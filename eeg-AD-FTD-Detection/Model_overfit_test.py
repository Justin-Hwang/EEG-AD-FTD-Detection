# Getting only one batch from our dataset to test whether the model could overfit

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
    LR          = 1e-3
    NUM_ITERS   = 500

    device = torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")

    # ─── 메타데이터 로드 ─────────────────────────────────
    with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
        all_meta = json.load(f)
    train_meta = [d for d in all_meta if d["type"] == "train"]

    # ─── 데이터셋 & 데이터로더 ───────────────────────────
    dataset    = EEGDataset(DATA_DIR, train_meta)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    # ─── 모델, 손실함수, 옵티마이저 ────────────────────────
    model = EEGformer(
        num_classes=3,
        in_channels=19,
        kernel_size=10,
        num_filters=120,
        rtm_blocks=3, stm_blocks=3, ttm_blocks=3,
        rtm_heads=3, stm_heads=3, ttm_heads=3,
        num_segments=15
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    # ─── 한 배치 가져오기 ─────────────────────────────────
    X_small, y_small = next(iter(dataloader))
    X_small, y_small = X_small.to(device), y_small.to(device)

    print(f"Overfit Test: {X_small.shape=} {y_small.shape=} on {device}")

    # ─── Overfit 훈련 루프 ─────────────────────────────────
    model.train()
    for i in range(1, NUM_ITERS+1):
        optimizer.zero_grad()
        logits = model(X_small)            # [BATCH_SIZE, 3]
        loss   = criterion(logits, y_small)
        loss.backward()
        optimizer.step()

        if i % 10 == 0 or i == 1:
            print(f"Iter {i:03d} | loss = {loss.item():.6f}")

    print("Finished overfit test.")

if __name__ == "__main__":
    main()
