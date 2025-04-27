"""
Load preprocessed EEG chunks and their labels into a PyTorch-compatible dataset.

This defines a custom PyTorch Dataset class that:

1. Loads .set files using MNE.
2. Extracts the EEG array: shape (channels, timepoints) = (19, 1425)
3. Returns a (data, label) pair for training.

This is plugged directly into a DataLoader for batching.
"""

import os
import mne
import warnings
from torch.utils.data import Dataset # Pytorch's base class for creating custom datasets

# Ignore RuntimeWarning
warnings.filterwarnings('ignore', category=RuntimeWarning)

# Enable CUDA
mne.utils.set_config('MNE_USE_CUDA', 'true')
mne.cuda.init_cuda(verbose=True)

# Reads a .set file using MNE
# Returns just the Numpy array.shape = (19 channels, 1425 timepoints)
def load_eeg_data(file_path):
    raw = mne.io.read_raw_eeglab(file_path)
    return raw.get_data()

# This class helps Pytorch Dataloader fetch EEG chunks and labels
class EEGDataset(Dataset):
    def __init__(self, data_directory, dataset):
        self.data_directory = data_directory # EEG File paths
        self.dataset = dataset # list of dic from labels.json 
        labels = [d['label'] for d in dataset]
        self.labels = labels
        data = [d['file_name'] for d in dataset]
        self.data = data

    # length of dataset
    def __len__(self):
        return len(self.dataset)

    # load one EEG sample + label
    def __getitem__(self, idx):
        file_info = self.dataset[idx]
        file_path = os.path.join(self.data_directory, file_info['file_name'])

        # Load raw EEG data using MNE
        # eeg_data = load_eeg_data(file_path)
        # eeg_data = eeg_data.astype('float32')
        eeg_data = load_eeg_data(file_path).astype('float32')
        # —— normalize each channel to zero mean, unit variance ——
        mean = eeg_data.mean(axis=1, keepdims=True)
        std  = eeg_data.std(axis=1, keepdims=True) + 1e-6
        eeg_data = (eeg_data - mean) / std
        
        # Label
        # A = 0
        # C = 1
        # F = 2
        label = 0 if file_info['label'] == 'A' else 1 if file_info['label'] == 'C' else 2
        # print(f'Label: {label} ({file_info["label"]})')

        return eeg_data, label
