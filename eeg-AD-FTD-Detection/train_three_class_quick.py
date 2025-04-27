# train_three_class_quick.py

import os
import json
import random
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from eeg_dataset import EEGDataset
from models import EEGformer

# ──────────────────────────────────────────────────────────
#  CONFIG
DATA_DIR    = "../model-data"
LABEL_FILE  = "labels.json"
BATCH_SIZE  = 16
LR          = 1e-4
EPOCHS      = 30
DEVICE      = torch.device("mps" if torch.backends.mps.is_available()
                           else "cuda" if torch.cuda.is_available()
                           else "cpu")

print(f"[INFO] Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────
# 1) Load & filter your three‐class training chunks
with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
    all_data = json.load(f)

train_data = [d for d in all_data if d["type"] == "train"]
labels     = [0 if d["label"]=="A"
              else 1 if d["label"]=="C"
              else 2 for d in train_data]

print("Raw train label counts:", Counter(labels))

# ──────────────────────────────────────────────────────────
# 2) Split into train/val (stratified)
train_idx, val_idx = train_test_split(
    list(range(len(train_data))),
    test_size=0.2,
    random_state=42,
    stratify=labels
)

train_subset = [train_data[i] for i in train_idx]
val_subset   = [train_data[i] for i in val_idx]
y_train      = [labels[i]     for i in train_idx]
y_val        = [labels[i]     for i in val_idx]

print("After split — train counts:", Counter(y_train))
print("After split —  val counts:", Counter(y_val))

# ──────────────────────────────────────────────────────────
# 3) Build datasets + compute sample weights
train_ds = EEGDataset(DATA_DIR, train_subset)
val_ds   = EEGDataset(DATA_DIR, val_subset)

# class frequencies in train
freq = Counter(y_train)
class_weights = {cls: 1.0/count for cls,count in freq.items()}
sample_weights = [ class_weights[y] for y in y_train ]
sampler = WeightedRandomSampler(sample_weights,
                                num_samples=len(sample_weights),
                                replacement=True)

train_loader = DataLoader(train_ds,
                          batch_size=BATCH_SIZE,
                          sampler=sampler)
val_loader   = DataLoader(val_ds,
                          batch_size=BATCH_SIZE,
                          shuffle=False)

# ──────────────────────────────────────────────────────────
# 4) Build model, loss, optimizer
model     = EEGformer(
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

# pass class_weights into CE
weights = torch.tensor([class_weights[i] for i in range(3)],
                       dtype=torch.float32, device=DEVICE)
criterion = nn.CrossEntropyLoss(weight=weights)
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ──────────────────────────────────────────────────────────
# 5) Training loop
for ep in range(1, EPOCHS+1):
    # — train —
    model.train()
    running_loss = 0.0
    correct = total = 0
    for X, y in train_loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * X.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds==y).sum().item()
        total   += y.size(0)

    train_loss = running_loss/total
    train_acc  = correct/total

    # — validate —
    model.eval()
    v_loss = v_correct = v_total = 0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for X, y in val_loader:
            X, y = X.to(DEVICE), y.to(DEVICE)
            logits = model(X)
            loss   = criterion(logits, y)

            v_loss += loss.item() * X.size(0)
            preds = logits.argmax(dim=1)
            v_correct += (preds==y).sum().item()
            v_total   += y.size(0)

            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(y.cpu().tolist())

    val_loss = v_loss/v_total
    val_acc  = v_correct/v_total

    print(f"Epoch {ep:02d} | "
          f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.3f} |  "
          f" Val Loss: {val_loss:.4f},  Val Acc: {val_acc:.3f}")

# ──────────────────────────────────────────────────────────
# 6) Final confusion matrix on val set
cm = confusion_matrix(all_labels, all_preds)
report = classification_report(all_labels, all_preds,
                               target_names=["A","C","F"], digits=4)
print("\nValidation Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)
