# test_quick_train.py
# To test the data integration works and check if there's any error in data pipeline

import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from eeg_dataset import EEGDataset
from models import EEGformer

# ─── CONFIG ─────────────────────────────────────────────────────────────
DATA_DIR    = "../model-data"
LABEL_FILE  = f"{DATA_DIR}/labels.json"
N_SAMPLES   = 300      # number of train-chunks to sample
BATCH_SIZE  = 16
EPOCHS      = 10
LR          = 1e-3
DEVICE      = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── LOAD & SAMPLE ────────────────────────────────────────────────────────
with open(LABEL_FILE, 'r') as f:
    all_data = json.load(f)
train_chunks = [d for d in all_data if d['type']=='train']

# down‐sample for speed and balance
sampled = random.sample(train_chunks, min(N_SAMPLES, len(train_chunks)))
labels = [0 if d['label']=='A' else 1 if d['label']=='C' else 2 for d in sampled]

# split into quick train/val
train_idx, val_idx = train_test_split(
    list(range(len(sampled))),
    test_size=0.2,
    stratify=labels,
    random_state=42
)

# build datasets & loaders
dataset = EEGDataset(DATA_DIR, sampled)
train_loader = DataLoader(Subset(dataset, train_idx), batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(Subset(dataset, val_idx),   batch_size=BATCH_SIZE, shuffle=False)

# ─── MODEL, LOSS, OPT ─────────────────────────────────────────────────────
model = EEGformer(
    num_classes=3,
    in_channels=19,
    kernel_size=10,
    num_filters=120,
    rtm_blocks=3, stm_blocks=3, ttm_blocks=3,
    rtm_heads=3,  stm_heads=3,  ttm_heads=3,
    num_segments=15
).to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ─── TRAIN & EVAL LOOP ────────────────────────────────────────────────────
for ep in range(1, EPOCHS+1):
    model.train()
    train_correct = train_total = 0
    train_loss = 0.0

    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * y.size(0)
        preds = logits.argmax(dim=1)
        train_correct += (preds == y).sum().item()
        train_total   += y.size(0)

    train_acc = train_correct / train_total
    avg_train_loss = train_loss / train_total

    model.eval()
    val_correct = val_total = 0
    val_loss = 0.0
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss = criterion(logits, y)
            val_loss += loss.item() * y.size(0)
            preds = logits.argmax(dim=1)
            val_correct += (preds == y).sum().item()
            val_total   += y.size(0)

    val_acc = val_correct / val_total
    avg_val_loss = val_loss / val_total

    print(f"Epoch {ep:02d} | "
          f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.3f} | "
          f" Val Loss: {avg_val_loss:.4f},  Val Acc: {val_acc:.3f}")

print("Done.")
