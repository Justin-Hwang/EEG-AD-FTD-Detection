import os
import json
import time
import gc
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedKFold

import optuna
import wandb

from eeg_dataset import EEGDataset
from models import EEGformer

# ─── Constants ─────────────────────────
LR_MIN, LR_MAX = 1e-5, 1e-4
WD_MIN, WD_MAX = 1e-5, 1e-3
FILTER_SIZE = 120  # Fixed filter size
HEAD_COUNT = 3
SEGMENT_CHOICES = [5, 15, 25]

N_FOLDS     = 5
MAX_EPOCHS  = 100
PATIENCE    = 15
BATCH_SIZE  = 16
NUM_WORKERS = max(1, min(4, os.cpu_count() - 1))
DATA_DIR    = "../model-data"
LABEL_FILE  = "labels.json"

DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    # Hyperparameter search space setup
    lr = trial.suggest_float("lr", LR_MIN, LR_MAX, log=True)
    weight_decay = trial.suggest_float("weight_decay", WD_MIN, WD_MAX, log=True)

    # Fixed values for num_filters and blocks
    num_filters = FILTER_SIZE  # Fixed filter size
    rtm_blocks = 3  # Fixed encoder block count
    stm_blocks = 3  # Fixed encoder block count
    ttm_blocks = 3  # Fixed encoder block count
    # Fixed Head Counts
    rtm_heads = HEAD_COUNT
    stm_heads = HEAD_COUNT
    ttm_heads = HEAD_COUNT
    num_segments = trial.suggest_categorical("num_segments", SEGMENT_CHOICES)

    # Pruning condition: only proceed if num_filters is divisible by heads
    for h in (rtm_heads, stm_heads, ttm_heads):
        if num_filters % h != 0:
            raise optuna.TrialPruned()

    # Initialize wandb
    wandb.init(project="eeg-cv-tuning-trial_10", config=trial.params)
    
    print(f"\n========================= Trial {trial.number} =========================")
    print(f"Testing with hyperparameters: {trial.params}")

    # Data preparation
    with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
        all_meta = json.load(f)
    train_meta = [d for d in all_meta if d["type"] == "train"]
    full_ds = EEGDataset(DATA_DIR, train_meta)
    labels = [d["label"] for d in train_meta]
    n_samples = len(full_ds)

    # StratifiedKFold setup
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    fold_metrics = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "best_epoch": []}

    for fold, (train_idx, val_idx) in enumerate(skf.split(range(n_samples), labels)):
        # Fold separation
        print(f"\n========================= Fold {fold} =========================")
        
        # Data loader setup
        train_loader = DataLoader(Subset(full_ds, train_idx), batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
        val_loader = DataLoader(Subset(full_ds, val_idx), batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)

        # Model and optimizer setup
        model = EEGformer(
            num_classes=3,
            in_channels=19,
            kernel_size=10,
            num_filters=num_filters,
            rtm_blocks=rtm_blocks,
            stm_blocks=stm_blocks,
            ttm_blocks=ttm_blocks,
            rtm_heads=rtm_heads,
            stm_heads=stm_heads,
            ttm_heads=ttm_heads,
            num_segments=num_segments
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        criterion = nn.CrossEntropyLoss()

        best_val_loss = float("inf")
        epochs_no_improve = 0
        best_epoch = 0
        best_train_l = best_train_a = best_val_a = None
        last_log_time = time.time()

        # Epoch-wise training
        for epoch in range(1, MAX_EPOCHS + 1):
            model.train()
            tl_sum = t_corr = t_tot = 0
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(X)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                tl_sum += loss.item()
                t_corr += (logits.argmax(1) == y).sum().item()
                t_tot += y.size(0)

            train_loss = tl_sum / len(train_loader)
            train_acc = t_corr / t_tot

            model.eval()
            vl_sum = v_corr = v_tot = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    logits = model(X)
                    loss = criterion(logits, y)
                    vl_sum += loss.item()
                    v_corr += (logits.argmax(1) == y).sum().item()
                    v_tot += y.size(0)

            val_loss = vl_sum / len(val_loader)
            val_acc = v_corr / v_tot

            step = fold * MAX_EPOCHS + epoch
            trial.report(val_loss, step=step)

            # Pruning check
            if trial.should_prune():
                print(f"\u274c Trial {trial.number} pruned at fold {fold}, epoch {epoch}")
                # Report the metrics before returning early
                for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc", "best_epoch"],
                                 [best_train_l, best_train_a, best_val_loss, best_val_a, best_epoch]):
                    fold_metrics[k].append(v)
                    trial.set_user_attr(f"fold{fold}_{k}", v)
                return  # End trial completely if pruned

            now = time.time()
            print(f"[Fold {fold}] Epoch {epoch:03d} | train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
                  f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | time={now - last_log_time:.1f}s")
            last_log_time = now

            # Early stopping: if validation loss does not improve
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch
                epochs_no_improve = 0
                best_train_l = train_loss
                best_train_a = train_acc
                best_val_a = val_acc
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= PATIENCE:
                    print(f"[Fold {fold}] Early stopping at epoch {epoch}, best was {best_epoch}")
                    # Report the metrics before returning early
                    for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc", "best_epoch"],
                                     [best_train_l, best_train_a, best_val_loss, best_val_a, best_epoch]):
                        fold_metrics[k].append(v)
                        trial.set_user_attr(f"fold{fold}_{k}", v)
                    return  # End trial completely if early stopping

        # Record results for the fold
        for k, v in zip(["train_loss", "train_acc", "val_loss", "val_acc", "best_epoch"],
                         [best_train_l, best_train_a, best_val_loss, best_val_a, best_epoch]):
            fold_metrics[k].append(v)
            trial.set_user_attr(f"fold{fold}_{k}", v)

        del model, optimizer, train_loader, val_loader
        torch.mps.empty_cache() if DEVICE.type == "mps" else torch.cuda.empty_cache()
        gc.collect()

    # Calculate average metrics
    avg = lambda k: sum(fold_metrics[k]) / N_FOLDS
    for key in ["train_loss", "train_acc", "val_loss", "val_acc", "best_epoch"]:
        trial.set_user_attr(f"avg_{key}", avg(key))

    wandb.finish()
    return avg("val_loss")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner(n_warmup_steps=1),
        study_name="eegformer_optuna_cv_3",
        storage="sqlite:///eegformer_optuna_cv_3.db",
        load_if_exists=True
    )
    study.optimize(objective, n_trials=10)  # Reduced trials to 10

    best = study.best_trial
    print("\n===== Best Trial Results =====")
    print(f"avg_val_loss   = {best.value:.6f}")
    print(f"avg_train_loss = {best.user_attrs['avg_train_loss']:.6f}")
    print(f"avg_train_acc  = {best.user_attrs['avg_train_acc']:.4f}")
    print(f"avg_val_acc    = {best.user_attrs['avg_val_acc']:.4f}")
    print(f"avg_best_epoch = {best.user_attrs['avg_best_epoch']:.1f}")
    print("best hyperparameters:")
    for k, v in best.params.items():
        print(f"  {k}: {v}")
    print("per-fold best metrics:")
    for f in range(N_FOLDS):
        print(
            f"  Fold {f}: epoch={best.user_attrs[f'fold{f}_best_epoch']}, "
            f"t_loss={best.user_attrs[f'fold{f}_train_loss']:.4f}, "
            f"t_acc={best.user_attrs[f'fold{f}_train_acc']:.4f}, "
            f"v_loss={best.user_attrs[f'fold{f}_val_loss']:.4f}, "
            f"v_acc={best.user_attrs[f'fold{f}_val_acc']:.4f}"
        )
