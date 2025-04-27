# train_kfold_eegformer.py

import os
import json
import random
import time
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, SubsetRandomSampler
from sklearn.model_selection import StratifiedKFold

from eeg_dataset import EEGDataset
from models import EEGformer
import utils       # for dscm_multiclass, plotting, etc.

# ──────────────────────────────────────────────────────────
# 0) Paths & seeds
DATA_DIR = "../model-data"
warnings.filterwarnings("ignore")
random.seed(42)
torch.manual_seed(42)

# ──────────────────────────────────────────────────────────
# 1) Create output dirs
os.makedirs("eegformer_models", exist_ok=True)
os.makedirs("eegformer_imgs",  exist_ok=True)

# ──────────────────────────────────────────────────────────
# 2) Hyperparameters
NUM_CHANS    = 19
TIMEPOINTS   = 1425
NUM_CLASSES  = 3

KERNEL_SIZE  = 10
NUM_FILTERS  = 120

NUM_BLOCKS   = 3
NUM_HEADS    = 3
NUM_SEGS     = 15

EPOCHS       = 50    # train longer
LR           = 1e-2   # start higher
BATCH_SIZE   = 20
WARMUP_EPOCHS = 5     # linear warm‑up

# ──────────────────────────────────────────────────────────
# 3) Device
DEVICE = torch.device(
    "mps"   if torch.backends.mps.is_available() else
    "cuda"  if torch.cuda.is_available()  else
    "cpu"
)
print(f"[INFO] Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────
# 4) Load + filter labels.json
with open(os.path.join(DATA_DIR, "labels.json"), "r") as f:
    all_data = json.load(f)
# only train‐type chunks
train_data = [d for d in all_data if d["type"] == "train"]

# ──────────────────────────────────────────────────────────
# 5) Build Dataset + label array
dataset = EEGDataset(DATA_DIR, train_data)
labels  = np.array([0 if d["label"] == "A" else
                    1 if d["label"] == "C" else
                    2
                    for d in train_data])

# ──────────────────────────────────────────────────────────
# 6) Class weights for imbalance
class_counts  = np.bincount(labels)               # e.g. [#A, #C, #F]
class_weights = 1.0 / class_counts
class_weights = torch.tensor(class_weights,
                             device=DEVICE,
                             dtype=torch.float32)

criterion = nn.CrossEntropyLoss(weight=class_weights)

# ──────────────────────────────────────────────────────────
# 7) 5‑fold CV
skf             = StratifiedKFold(n_splits=5,
                                  shuffle=True,
                                  random_state=42)
all_fold_losses = []

for fold, (train_idx, val_idx) in enumerate(
        skf.split(np.zeros(len(labels)), labels), start=1):
    print(f"\n=== Fold {fold}/5 ===")
    train_sampler = SubsetRandomSampler(train_idx)
    val_sampler   = SubsetRandomSampler(val_idx)

    train_loader = DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              sampler=train_sampler)
    val_loader   = DataLoader(dataset,
                              batch_size=BATCH_SIZE,
                              sampler=val_sampler)

    # fresh model each fold
    model = EEGformer(
        num_classes=NUM_CLASSES,
        in_channels=NUM_CHANS,
        kernel_size=KERNEL_SIZE,
        num_filters=NUM_FILTERS,
        rtm_blocks=NUM_BLOCKS,
        stm_blocks=NUM_BLOCKS,
        ttm_blocks=NUM_BLOCKS,
        rtm_heads=NUM_HEADS,
        stm_heads=NUM_HEADS,
        ttm_heads=NUM_HEADS,
        num_segments=NUM_SEGS
    ).to(DEVICE)

    optimizer = optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=0.1,    # begin at 0.1×LR
        end_factor=1.0,      # ramp to full LR
        total_iters=WARMUP_EPOCHS
    )

    fold_losses = []
    for ep in range(1, EPOCHS + 1):
        t0 = time.time()
        model.train()
        running_loss = 0.0

        for X, y in train_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(X)
            loss   = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        fold_losses.append(avg_train_loss)

        # warm‑up scheduler
        scheduler.step()

        # validation
        model.eval()
        correct = total = 0
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(DEVICE), y.to(DEVICE)
                preds = model(X).argmax(dim=1)
                correct += (preds == y).sum().item()
                total   += y.size(0)
        val_acc = correct / total

        print(f"Fold {fold}, Epoch {ep}/{EPOCHS}  "
              f"train_loss={avg_train_loss:.4f}  "
              f"val_acc={val_acc:.4f}  "
              f"time={time.time()-t0:.1f}s  "
              f"LR:{optimizer.param_groups[0]['lr']:.2e}")

    # save this fold
    torch.save(model.state_dict(),
               f"eegformer_models/eegformer_5fold_fold{fold}.pth")
    plt.plot(fold_losses, label=f"Fold {fold}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"eegformer_imgs/train_loss_fold{fold}.png")
    plt.close()

    all_fold_losses += fold_losses

# combined‐fold loss curve
plt.plot(all_fold_losses)
plt.title("All Folds Training Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("eegformer_imgs/train_losses_5fold_all.png")
plt.close()

print("\n✅ EEGformer 5‑fold training complete.")
