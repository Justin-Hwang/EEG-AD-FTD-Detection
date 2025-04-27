# inference_eegformer.py

import json
import torch
from torch.utils.data import DataLoader
from eeg_dataset import EEGDataset
from models import EEGformer
import utils

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Load test data (test_cross only)
with open("model-data/labels.json", 'r') as f:
    data_info = json.load(f)
test_data = [d for d in data_info if d['type'] == 'test_cross']
dataset = EEGDataset("model-data", test_data)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Load trained model
model_path = "eegformer_models/eegformer_final.pth"
model = EEGformer(torch.zeros(1425, 19), 3, 19, 10, 3, 3, 3, 3, 15, 6)
model.load_state_dict(torch.load(model_path, map_location=device))
model.to(device)
model.eval()
print(f"Loaded model from {model_path}")

# Evaluation
y_true, y_pred = [], []
with torch.no_grad():
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        output = model(x)
        pred = torch.argmax(output, dim=1)
        y_true.append(y.item())
        y_pred.append(pred.item())

# Print multi-class evaluation
utils.dscm_multiclass(y_pred, y_true)
