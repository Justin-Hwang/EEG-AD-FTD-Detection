# Model_overfit_test_optuna.py
# Previous Trial

import os
import json
import multiprocessing

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import optuna
import wandb

from eeg_dataset import EEGDataset
from models import EEGformer

# ─── 탐색 공간 정의 ─────────────────────────────────────────
LR_MIN, LR_MAX = 1e-5, 1e-1
WD_MIN, WD_MAX = 1e-6, 1e-2
FILTER_MIN, FILTER_MAX = 60, 240
FILTER_STEP = 30

HEAD_CHOICES    = [1, 2, 4, 8, 12]
BLOCK_CHOICES   = [1, 2, 3, 4, 5]
SEGMENT_CHOICES = [5, 10, 15, 20, 25]

BATCH_SIZE = 16
NUM_ITERS  = 200
DATA_DIR   = "../model-data"
LABEL_FILE = "labels.json"

def objective(trial):
    # 1) 하이퍼파라미터 샘플링
    lr           = trial.suggest_float("lr", LR_MIN, LR_MAX, log=True)
    weight_decay = trial.suggest_float("weight_decay", WD_MIN, WD_MAX, log=True)
    num_filters  = trial.suggest_int("num_filters", FILTER_MIN, FILTER_MAX, step=FILTER_STEP)
    rtm_blocks   = trial.suggest_categorical("rtm_blocks", BLOCK_CHOICES)
    stm_blocks   = trial.suggest_categorical("stm_blocks", BLOCK_CHOICES)
    ttm_blocks   = trial.suggest_categorical("ttm_blocks", BLOCK_CHOICES)
    rtm_heads    = trial.suggest_categorical("rtm_heads", HEAD_CHOICES)
    stm_heads    = trial.suggest_categorical("stm_heads", HEAD_CHOICES)
    ttm_heads    = trial.suggest_categorical("ttm_heads", HEAD_CHOICES)
    num_segments = trial.suggest_categorical("num_segments", SEGMENT_CHOICES)

    # embed_dim=num_filters 가 heads로 나누어떨어져야 함
    for h in (rtm_heads, stm_heads, ttm_heads):
        if num_filters % h != 0:
            raise optuna.TrialPruned()

    # 2) 모델·디바이스·손실·옵티마이저 설정
    device = torch.device("mps" if torch.backends.mps.is_available()
                        else "cuda" if torch.cuda.is_available()
                        else "cpu")
    model = EEGformer(
        num_classes=3,
        in_channels=19,
        kernel_size=10,
        num_filters=num_filters,
        rtm_blocks=rtm_blocks,
        stm_blocks=stm_blocks,
        ttm_blocks=ttm_blocks,
        rtm_heads=rtm_heads,
        stm_heads=stm_heads,
        ttm_heads=ttm_heads,
        num_segments=num_segments
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    # 3) 데이터 로드 & 한 배치 추출
    with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
        all_meta = json.load(f)
    train_meta = [d for d in all_meta if d["type"] == "train"]
    dataset    = EEGDataset(DATA_DIR, train_meta)
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,      # spawn 문제 회피
        pin_memory=True
    )
    X_small, y_small = next(iter(dataloader))
    X_small, y_small = X_small.to(device), y_small.to(device)

    # 4) Overfit 훈련 루프
    model.train()
    for i in range(1, NUM_ITERS + 1):
        optimizer.zero_grad()
        logits = model(X_small)            # [BATCH_SIZE, num_classes]
        loss   = criterion(logits, y_small)
        loss.backward()
        optimizer.step()

        # accuracy 계산
        preds = logits.argmax(dim=1)
        acc = (preds == y_small).float().mean().item()

        # Optuna intermediate 보고 (pruning)
        trial.report(loss.item(), step=i)
        if trial.should_prune():
            raise optuna.TrialPruned()

        # 터미널 출력, WandB 로깅
        if i == 1 or i % 10 == 0:
            print(f"Iter {i:03d} | loss = {loss.item():.6f} | acc = {acc:.4f}")
        wandb.log({"overfit/loss": loss.item(), "overfit/acc": acc}, step=i)

    return loss.item()


if __name__ == "__main__":
    # spawn-safe
    multiprocessing.freeze_support()

    # WandB init
    wandb.init(
        project="eeg-overfit-tuning",
        reinit=True,
        config={
            "batch_size": BATCH_SIZE,
            "num_iters": NUM_ITERS
        }
    )

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(),
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=20)

    print("Best trial parameters:", study.best_trial.params)
    print("Best trial loss   :", study.best_trial.value)
