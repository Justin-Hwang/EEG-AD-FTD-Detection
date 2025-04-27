import json
import torch
from torch.utils.data import DataLoader
from eeg_dataset import EEGDataset
from models import EEGformer

def main():
    # 1) Pick your device
    device = torch.device("cuda" if torch.cuda.is_available() else 
                          "mps"   if torch.backends.mps.is_available() else 
                          "cpu")
    print(f"[INFO] Testing on device: {device}")

    # 2) Load a small subset of your train data
    with open("../model-data/labels.json", "r") as f:
        all_data = json.load(f)
    train_data = [d for d in all_data if d["type"] == "train"]
    # just take first 8 samples so this runs instantly
    subset = train_data[:8]

    # 3) Build DataLoader
    dataset = EEGDataset("../model-data", subset)
    loader  = DataLoader(dataset, batch_size=4, shuffle=False)

    # 4) Instantiate your model
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
    ).to(device)

    # 5) Grab one batch and run it through
    X, y = next(iter(loader))
    X, y = X.to(device), y.to(device)
    print(f"Input batch X.shape = {X.shape}, y.shape = {y.shape}")

    # forward
    logits = model(X)
    print(f"Output logits.shape = {logits.shape}")

    # 6) Try a dummy loss & backward
    criterion = torch.nn.CrossEntropyLoss()
    loss = criterion(logits, y)
    print(f"Dummy cross‐entropy loss = {loss.item():.4f}")

    loss.backward()
    print("✅ Forward & backward pass succeeded.")

if __name__ == "__main__":
    main()
