import mne
import matplotlib.pyplot as plt

print("MNE version:", mne.__version__)

# Load the EEG file
file_path = "model-data/test/sub-001_eeg_chunk_23.set"
raw = mne.io.read_raw_eeglab(file_path, preload=True)

print(raw.info)

# Plot EEG
raw.plot(n_channels=19, title="EEG Data")

# Required on macOS to show plots sometimes
plt.show()
