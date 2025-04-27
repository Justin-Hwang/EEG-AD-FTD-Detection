import os
import json
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from eeg_dataset import EEGDataset  # Make sure to import your actual dataset class
from models import ODCM, RTM, STM, TTM, CNNDecoder

# Directory to save the plots
SAVE_DIR = "./intermediate_output_vis"  # Set to your desired directory
os.makedirs(SAVE_DIR, exist_ok=True)

# === Visualization and Dimension Tracking ===
def plot_intermediate_output(output_dict, batch_idx):
    for stage, output in output_dict.items():
        plt.figure(figsize=(10, 5))
        
        if stage == 'output':  # Decoder output is different
            # For the decoder output (final output), it's a 1D vector with class scores
            output = output[0].cpu().detach().numpy()  # Take the first batch item for visualization
            plt.title(f"Output at {stage} - Predicted Class")
            plt.bar(range(len(output)), output)  # Visualizing as a bar plot for class scores
            plt.xlabel('Class')
            plt.ylabel('Score')
        else:
            # For intermediate outputs like ODCM, RTM, STM, and TTM, visualize as images
            output = output[0].cpu().detach().numpy()  # Get the first element of the batch
            plt.title(f"Output at {stage}")
            plt.imshow(output, cmap="viridis", aspect="auto")
            plt.colorbar()
            plt.xlabel('Time Steps')
            plt.ylabel('Channels / Features')

        # Save the figure to the disk
        save_path = os.path.join(SAVE_DIR, f"batch_{batch_idx}_{stage}.png")
        plt.savefig(save_path)  # Save the figure as a PNG file
        plt.close()  # Close the figure to free memory

# Define the EEGformer model with dimension tracking
class EEGformerWithTracking(nn.Module):
    def __init__(self, num_classes, in_channels, kernel_size, num_filters,
                 rtm_blocks, stm_blocks, ttm_blocks,
                 rtm_heads, stm_heads, ttm_heads, num_segments):
        super().__init__()

        self.odcm = ODCM(in_channels, kernel_size, num_filters)
        self.rtm = RTM(num_filters, rtm_heads, rtm_blocks)
        self.stm = STM(num_filters, stm_heads, stm_blocks)
        self.ttm = TTM(num_segments, num_filters, ttm_heads, ttm_blocks)
        self.decoder = CNNDecoder(num_filters, num_classes)

    def forward(self, x):  # [B, 19, 1425]
        intermediate_outputs = {}

        # 1D CNN output
        x1 = self.odcm(x)   # [B, 120, T]
        intermediate_outputs['odcm'] = x1
        print(f"ODCM Output Shape: {x1.shape}")
        
        # Regional Transformer output
        x2 = self.rtm(x1)
        intermediate_outputs['rtm'] = x2
        print(f"RTM Output Shape: {x2.shape}")

        # Synchronous Transformer output
        x3 = self.stm(x2)
        intermediate_outputs['stm'] = x3
        print(f"STM Output Shape: {x3.shape}")

        # Temporal Transformer output
        x4 = self.ttm(x3)    # [B, M, 120]
        intermediate_outputs['ttm'] = x4
        print(f"TTM Output Shape: {x4.shape}")

        # Output before the decoder (Encoder output)
        encoder_output = x4
        intermediate_outputs['encoder_output'] = encoder_output
        print(f"Encoder Output Shape (Before Decoder): {encoder_output.shape}")

        # Decoder output
        output = self.decoder(encoder_output)  # [B, num_classes]
        intermediate_outputs['output'] = output
        print(f"Decoder Output Shape (Final Output): {output.shape}")

        return intermediate_outputs

# Main function for running the model with real training data
def main():
    # Set up device and dataset
    DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")

    DATA_DIR = "../model-data"  # Make sure to adjust this path accordingly
    LABEL_FILE = "labels.json"  # Path to your label file

    # Load your dataset
    with open(os.path.join(DATA_DIR, LABEL_FILE), "r") as f:
        all_meta = json.load(f)
    train_meta = [d for d in all_meta if d["type"] == "train"]
    
    # Create your dataset instance
    full_ds = EEGDataset(DATA_DIR, train_meta)
    
    # Define DataLoader for batching
    train_loader = DataLoader(full_ds, batch_size=32, shuffle=True, num_workers=4)
    
    # Initialize the model with fixed parameters
    model = EEGformerWithTracking(
        num_classes=3,  # Adjust the number of classes according to your dataset
        in_channels=19,  # Channels in the EEG data (adjust accordingly)
        kernel_size=10,  # Your desired kernel size
        num_filters=120,  # Fixed filter size
        rtm_blocks=3,  # Fixed blocks for the encoder
        stm_blocks=3,
        ttm_blocks=3,
        rtm_heads=3,  # Fixed number of heads
        stm_heads=3,
        ttm_heads=3,
        num_segments=15  # Number of segments
    ).to(DEVICE)

    # Run only for the first batch
    for batch_idx, (X, y) in enumerate(train_loader):
        if batch_idx == 0:  # Only process the first batch
            X = X.to(DEVICE)  # Move data to the device (GPU/CPU)
            
            # Forward pass to get the outputs at each stage
            output_dict = model(X)
            
            # Visualize and save the intermediate outputs
            plot_intermediate_output(output_dict, batch_idx)
            # break

if __name__ == "__main__":
    main()
