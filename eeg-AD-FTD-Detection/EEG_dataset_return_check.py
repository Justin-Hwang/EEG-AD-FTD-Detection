# Test the EEG dataset module could get the every data in a correct format

import os
import json
from collections import Counter

import torch
from eeg_dataset import EEGDataset

# ──────────────────────────────────────────────
DATA_DIR   = "../model-data"
LABEL_FILE = "labels.json"

# ──────────────────────────────────────────────
# 1) Load all “train” metadata
with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
    all_meta = json.load(f)
train_meta = [d for d in all_meta if d["type"] == "train"]

# 2) Instantiate the FULL dataset
dataset = EEGDataset(DATA_DIR, train_meta)

# 3) Inspect every sample
print(f"Dataset size: {len(dataset)} samples\n")
all_labels = []
for i in range(len(dataset)):
    X, y = dataset[i]
    all_labels.append(int(y))

    if isinstance(X, torch.Tensor):
        mn, mx = float(X.min()), float(X.max())
        print(f"{i:04d} | X.shape={tuple(X.shape)} | dtype={X.dtype} | min={mn:.4f} | max={mx:.4f} | y={int(y)}")
    else:
        # adapt this if your dataset returns numpy arrays or something else
        arr = X if hasattr(X, 'shape') else torch.as_tensor(X)
        mn, mx = float(arr.min()), float(arr.max())
        print(f"{i:04d} | X.shape={tuple(arr.shape)} | min={mn:.4f} | max={mx:.4f} | y={y}")

# 4) Finally, show overall label distribution
print("\nLabel distribution across entire training set:")
print(Counter(all_labels))
