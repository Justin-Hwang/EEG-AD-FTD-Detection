# train_three_class_kfold_wandb.py

import os
import json
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import wandb

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold
from eeg_dataset import EEGDataset
from models import EEGformer


def train():
    # Initialize W&B run
    wandb.init()
    cfg = wandb.config

    # Device
    DEVICE = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    print(f"[INFO] Using device: {DEVICE}")

    # Load & filter labels
    with open(os.path.join(cfg.data_dir, cfg.label_file), "r") as f:
        all_data = json.load(f)
    train_data = [d for d in all_data if d["type"] == "train"]
    labels = np.array([
        0 if d["label"] == "A" else
        1 if d["label"] == "C" else
        2
        for d in train_data
    ], dtype=int)

    print(f"[INFO] Found {len(train_data)} train‑type samples, "
          f"label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    # Class weights to handle imbalance
    class_counts = np.bincount(labels, minlength=3)
    cw = 1.0 / class_counts
    class_weights = torch.tensor(cw, dtype=torch.float32, device=DEVICE)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # 5‑fold CV
    skf = StratifiedKFold(n_splits=cfg.n_splits, shuffle=True, random_state=42)
    for fold, (train_idx, val_idx) in enumerate(
            skf.split(np.zeros(len(labels)), labels), start=1):
        print(f"\n=== Fold {fold}/{cfg.n_splits} ===")
        fold_start = time.time()

        # DataLoaders
        dataset = EEGDataset(cfg.data_dir, train_data)
        train_loader = DataLoader(
            dataset, batch_size=cfg.batch_size,
            sampler=SubsetRandomSampler(train_idx)
        )
        val_loader = DataLoader(
            dataset, batch_size=cfg.batch_size,
            sampler=SubsetRandomSampler(val_idx)
        )

        # Model, optimizer, scheduler
        model = EEGformer(
            num_classes=3,
            in_channels=19,
            kernel_size=cfg.kernel_size,
            num_filters=cfg.num_filters,
            rtm_blocks=cfg.num_blocks,
            stm_blocks=cfg.num_blocks,
            ttm_blocks=cfg.num_blocks,
            rtm_heads=cfg.num_heads,
            stm_heads=cfg.num_heads,
            ttm_heads=cfg.num_heads,
            num_segments=cfg.num_segments
        ).to(DEVICE)

        optimizer = optim.AdamW(model.parameters(), lr=cfg.lr)
        scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=cfg.warmup_start_factor,
            end_factor=1.0,
            total_iters=cfg.warmup_epochs
        )

        # watch gradients & weights
        wandb.watch(model, log="all", log_freq=cfg.log_freq)

        # Epoch loop
        for ep in range(1, cfg.epochs + 1):
            ep_start = time.time()

            # — Train —
            model.train()
            tot_loss = tot_corr = tot_samples = 0
            for X, y in train_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                optimizer.zero_grad()
                logits = model(X)
                loss   = criterion(logits, y)
                loss.backward()
                optimizer.step()

                preds       = logits.argmax(dim=1)
                tot_loss   += loss.item() * X.size(0)
                tot_corr   += (preds == y).sum().item()
                tot_samples += X.size(0)

            train_loss = tot_loss / tot_samples
            train_acc  = tot_corr / tot_samples

            # — Validate —
            model.eval()
            v_loss = v_corr = v_tot = 0
            with torch.no_grad():
                for X, y in val_loader:
                    X, y = X.to(DEVICE), y.to(DEVICE)
                    logits = model(X)
                    loss   = criterion(logits, y)

                    preds    = logits.argmax(dim=1)
                    v_loss   += loss.item() * X.size(0)
                    v_corr   += (preds == y).sum().item()
                    v_tot    += X.size(0)

            val_loss = v_loss / v_tot
            val_acc  = v_corr / v_tot

            # — Time tracking —
            ep_time = time.time() - ep_start

            # Log & print
            print(f"Fold {fold} Ep {ep:02d} | "
                  f"tr_loss={train_loss:.4f} tr_acc={train_acc:.3f} | "
                  f"va_loss={val_loss:.4f} va_acc={val_acc:.3f} | "
                  f"time={ep_time:.1f}s")

            wandb.log({
                f"fold{fold}/train_loss":  train_loss,
                f"fold{fold}/train_acc":   train_acc,
                f"fold{fold}/val_loss":    val_loss,
                f"fold{fold}/val_acc":     val_acc,
                f"fold{fold}/epoch_time":  ep_time,
            }, step=ep)

            scheduler.step()

        # Fold total time
        fold_time = time.time() - fold_start
        print(f"--- Fold {fold} completed in {fold_time:.1f}s ---")
        # push into W&B summary
        wandb.summary[f"fold{fold}_total_time"] = fold_time

        # save model for this fold
        os.makedirs("eegformer_models", exist_ok=True)
        torch.save(
            model.state_dict(),
            f"eegformer_models/eegformer_fold{fold}.pth"
        )

    wandb.finish()


if __name__ == "__main__":
    random.seed(42)
    torch.manual_seed(42)

    # Sweep configuration
    sweep_config = {
      "method": "bayes",
      "metric": {"name": "fold1/val_acc", "goal": "maximize"},
      "parameters": {
        "lr": {
          "distribution": "log_uniform_values",
          "min": 1e-5,
          "max": 5e-4
        },
        "batch_size": {"values": [16, 32]},
        "epochs":     {"value": 30},
        "n_splits":   {"value": 5},
        "num_blocks":   {"values": [2, 3, 4]},
        "num_filters":  {"values": [60, 120, 180]},
        "num_heads":    {"values": [2, 3, 4]},
        "num_segments": {"values": [10, 15, 20]},
        # warm‑up
        "warmup_epochs":       {"value": 5},
        "warmup_start_factor": {"value": 0.1},
        # fixed model hyperparams
        "kernel_size":  {"value": 10},
        "label_file":   {"value": "labels.json"},
        "data_dir":     {"value": "../model-data"},
        "log_freq":     {"value": 10}
      }
    }

    sweep_id = wandb.sweep(sweep_config, project="EEG_KFold_Tuning")
    wandb.agent(sweep_id, function=train, count=20)
