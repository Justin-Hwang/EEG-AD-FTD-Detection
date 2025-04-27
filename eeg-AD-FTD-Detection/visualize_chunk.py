import mne
import os

# 🔁 Path to your EEG chunk file (15-second window)
chunk_path = "model-data/train/sub-077_eeg_chunk_0.set"  # Update if needed

# ✅ Load the EEG chunk
raw = mne.io.read_raw_eeglab(chunk_path, preload=True)

# 📄 Show info
print("\nEEG Chunk Info:")
print(raw.info)

# 🧠 Add standard 10-20 electrode positions (if not already included)
raw.set_montage("standard_1020")

# ✅ Plot time-domain EEG signal
raw.plot(n_channels=19, scalings='auto', title="EEG Chunk - sub-077", show=True)

# ✅ Plot Power Spectral Density (frequency domain)
raw.plot_psd(fmin=1, fmax=45, average=True, show=True)

# ✅ Plot Topomap (distribution of frequency power across scalp)
raw.compute_psd(fmin=8, fmax=12).plot_topomap(ch_type='eeg', normalize=True, show=True)
