# test_overfit_subset.py

import json
import torch
import random
from torch.utils.data import DataLoader, SubsetRandomSampler
from eeg_dataset import EEGDataset
from models import EEGformer
import torch.nn as nn
import torch.optim as optim

# ──────────────────────────────────────────────────────────
# 1) Settings
DATA_DIR      = "../model-data"
LABELS_FILE   = f"{DATA_DIR}/labels.json"
SUBSET_SIZE   = 20      # number of chunks to overfit
BATCH_SIZE    = 4
EPOCHS        = 50
LR            = 1e-3
DEVICE = torch.device(
    "mps"   if torch.backends.mps.is_available() else
    "cuda"  if torch.cuda.is_available()  else
    "cpu"
)
print(f"[INFO] Using device: {DEVICE}")

# ──────────────────────────────────────────────────────────
# 2) Load and select a tiny subset of the train data
with open(LABELS_FILE, "r") as f:
    all_chunks = json.load(f)

train_chunks = [c for c in all_chunks if c["type"] == "train"]
subset_chunks = train_chunks[:SUBSET_SIZE]

dataset = EEGDataset(DATA_DIR, subset_chunks)
indices = list(range(len(dataset)))
sampler = SubsetRandomSampler(indices)
loader  = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=sampler)

# ──────────────────────────────────────────────────────────
# 3) Instantiate your model on GPU/CPU
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
optimizer = optim.AdamW(model.parameters(), lr=LR)

# ──────────────────────────────────────────────────────────
# 4) Training loop—watch the loss go down!
print(f"Overfitting on {SUBSET_SIZE} samples, batch size {BATCH_SIZE}")
for epoch in range(1, EPOCHS+1):
    model.train()
    running_loss = 0.0

    for X, y in loader:
        X, y = X.to(DEVICE), y.to(DEVICE)
        optimizer.zero_grad()
        logits = model(X)            # [BATCH_SIZE, 3]
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    avg_loss = running_loss / len(loader)
    print(f"Epoch {epoch:02d} | loss = {avg_loss:.4f}")

print("Done. If loss → ~0.0, your model & data pipeline are working correctly!")
