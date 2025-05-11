import os
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from eeg_dataset import EEGDataset
from models import EEGformer

def main():
    # ─── 설정 ──────────────────────────────────────────
    DATA_DIR    = "../model-data"
    LABEL_FILE  = "labels.json"
    BATCH_SIZE  = 16
    LR          = 5e-5
    NUM_ITERS   = 100
    KERNEL_SIZE = 10      # ODCM Kernel Size
    NUM_FILTERS = 120     # ODCM Filter (C)
    NUM_HEADS   = 3       # Transformer Heads
    NUM_BLOCKS  = 3       # Transformer Blocks
    NUM_SEGMENTS= 15      # TTM time segments (M)
    NUM_CLASSES = 3       # Class

    device = torch.device(
        "mps" if torch.backends.mps.is_available() else
        "cuda" if torch.cuda.is_available() else
        "cpu"
    )

    # ─── Meta Data Load─────────────────────────────────
    with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
        all_meta = json.load(f)
    train_meta = [d for d in all_meta if d["type"] == "train"]

    # ─── Dataset & DataLoader ───────────────────────────
    dataset    = EEGDataset(DATA_DIR, train_meta)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ─── Extract One Batch ─────────────────────────────────
    X_small, y_small = next(iter(dataloader))
    X_small, y_small = X_small.to(device), y_small.to(device)
    B, S, L = X_small.shape  # Batch, Channels, Time-length

    print(f"Overfit Test Batch Shape: X_small={X_small.shape}, y_small={y_small.shape} on {device}")

    # ─── Model, Loss, Optimizer ────────────────────────
    model = EEGformer(
        in_channels  = S,
        input_length = L,
        kernel_size  = KERNEL_SIZE,
        num_filters  = NUM_FILTERS,
        num_heads    = NUM_HEADS,
        num_blocks   = NUM_BLOCKS,
        num_segments = NUM_SEGMENTS,
        num_classes  = NUM_CLASSES,
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ─── Overfit 훈련 루프 ─────────────────────────────────
    model.train()
    for i in range(1, NUM_ITERS+1):
        optimizer.zero_grad()
        logits = model(X_small)            # [BATCH_SIZE, NUM_CLASSES]
        loss   = criterion(logits, y_small)
        loss.backward()
        optimizer.step()

        if i == 1 or i % 10 == 0:
            # 배치 정확도 계산
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                acc   = (preds == y_small).float().mean().item() * 100
            print(f"Iter {i:03d} | loss = {loss.item():.6f} | acc = {acc:5.2f}%")

    print("Finished overfit test.")

if __name__ == "__main__":
    main()
