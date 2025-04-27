# Sanity check of EEG Dataset Module
# Test whether if there's any error or missing values in EEG Dataset Module

import os
import json
from collections import Counter

import torch
from eeg_dataset import EEGDataset

# ──────────────────────────────────────────────
DATA_DIR   = "../model-data"
LABEL_FILE = "labels.json"

# 1) Load all “train” metadata
with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
    all_meta = json.load(f)
train_meta = [d for d in all_meta if d["type"] == "train"]

# 2) Instantiate the dataset
dataset = EEGDataset(DATA_DIR, train_meta)

# 3) Length check
print(f"Number of train samples: {len(dataset)}\n")

# 4) Sample shape & label check at specific indices
indices = [0, len(dataset)//2, len(dataset)-1]
for idx in indices:
    X, y = dataset[idx]
    arr = X if isinstance(X, torch.Tensor) else torch.as_tensor(X)
    mn, mx = float(arr.min()), float(arr.max())
    print(f"Index {idx:04d} | shape={tuple(arr.shape)} | dtype={arr.dtype} | min={mn:.4f} | max={mx:.4f} | label={y}")

# 5) Overall label distribution
all_labels = [int(dataset[i][1]) for i in range(len(dataset))]
print("\nLabel distribution across entire training set:")
print(Counter(all_labels))
