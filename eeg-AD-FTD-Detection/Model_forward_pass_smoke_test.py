# 모델 훈련 전에 전체 데이터 로딩 및 모델의 입력/출력 흐름이 정상인지 확인하는 "forward-pass smoke test"**입니다.
# "모델에 데이터를 한 번 넣어보고, 아무 문제 없이 돌아가는지 확인하자!"
# Before we actually train the model, we should tesk whether the dataset is imported correctly to the model
# Then, we should test whether the model could get the correct input and produce the expected output
# This is called smoke test

import os
import json

import torch
from torch.utils.data import DataLoader
from eeg_dataset import EEGDataset
from models import EEGformer

# ──────────────────────────────────────────────
DATA_DIR    = "../model-data"
LABEL_FILE  = "labels.json"
BATCH_SIZE  = 16
NUM_WORKERS = 0   # macOS/Windows에서는 0으로 두는 게 안전합니다

def main():
    # 1) Load all “train” metadata
    with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
        all_meta = json.load(f)
    train_meta = [d for d in all_meta if d["type"] == "train"]

    # 2) Instantiate the FULL dataset and wrap in a DataLoader
    dataset    = EEGDataset(DATA_DIR, train_meta)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    print(f"Number of train samples: {len(dataset)}")
    print(f"Batch size: {BATCH_SIZE}\n")

    # 3) Instantiate the model and move to device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")
    model  = EEGformer(
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
        num_segments=15,
    ).to(device)

    model.eval()

    # 4) Grab one batch and do a forward‐pass smoke test
    X_batch, y_batch = next(iter(dataloader))
    X_batch = X_batch.to(device)
    y_batch = y_batch.to(device)

    with torch.no_grad():
        outputs = model(X_batch)

    # 5) Print shapes and a few sample outputs
    print(f"Input batch shape : {tuple(X_batch.shape)}")
    print(f"Output batch shape: {tuple(outputs.shape)}")
    print("Sample logits:\n", outputs[:3])
    print("Predicted classes:", outputs.argmax(dim=1)[:3].cpu().tolist())
    print("True labels      :", y_batch[:3].cpu().tolist())


if __name__ == "__main__":
    main()
