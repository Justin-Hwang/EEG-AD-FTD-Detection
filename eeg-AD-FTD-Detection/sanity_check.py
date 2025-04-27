#!/usr/bin/env python3
import os
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from eeg_dataset import EEGDataset   # your dataset
from models import EEGformer         # your model

# ──────────────────────────────────────────────
# 0) Deterministic setup for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark     = False
torch.use_deterministic_algorithms(True)

# ──────────────────────────────────────────────
# CONFIG
DATA_DIR    = "../model-data"
LABEL_FILE  = "labels.json"
SUBSET_SIZE = 20    # tiny number to really force overfit
EPOCHS      = 100
LR          = 0.01  
BATCH_SIZE  = 5

DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
print(f"[INFO] Device: {DEVICE}")

# ──────────────────────────────────────────────
# 1) Load your metadata and filter to training‐only
with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
    all_meta = json.load(f)
train_meta = [d for d in all_meta if d["type"] == "train"]

# ──────────────────────────────────────────────
# 2) Randomly pick a small subset *of the metadata*,
#    so you’ll definitely see a mix of labels in that subset
subset_meta = random.sample(train_meta, SUBSET_SIZE)

# 3) Build a dataset on that tiny subset, then a DataLoader
dataset = EEGDataset(DATA_DIR, subset_meta)
loader  = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,       # simple shuffle in DataLoader
    num_workers=0,      # no extra workers for total determinism
    pin_memory=True
)

# 4) Build model, loss, optimizer
model = EEGformer(
    num_classes=3,
    in_channels=19,
    kernel_size=10,
    num_filters=120,
    rtm_blocks=3,
    stm_blocks=3,
    ttm_blocks=3,
    rtm_heads=3,
    stm_heads=3,
    ttm_heads=3,
    num_segments=15
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.0)

# ──────────────────────────────────────────────
# 5) Training loop on the tiny subset, with debug
first_batch = True
for ep in range(1, EPOCHS + 1):
    model.train()
    total_loss = 0.0
    correct    = 0
    total      = 0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)

        # Print out the very first batch so you can inspect labels
        if first_batch:
            print("DEBUG batch X.shape:", X.shape)
            print("DEBUG batch y:", y.tolist())
            first_batch = False

        optimizer.zero_grad()
        logits = model(X)             # raw scores from your model
        loss   = criterion(logits, y) # CrossEntropyLoss expects raw logits
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        total_loss += loss.item() * X.size(0)
        preds       = logits.argmax(dim=1)
        correct    += (preds == y).sum().item()
        total      += X.size(0)

    avg_loss = total_loss / total
    acc      = correct / total
    print(f"[Epoch {ep:03d}] loss = {avg_loss:.4f} | acc = {acc:.3f}")

    # Stop once it's memorized perfectly
    if avg_loss < 1e-4 and acc > 0.99:
        print("🎉 Overfit achieved — sanity check passed!")
        break

print("Done.")
