import torch
import torch.nn as nn
import math

# === Basic Transformer Block ===
class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_output, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        x = x + self.mlp(self.norm2(x))
        return x

# === Depthwise Temporal Convolution ===
class ODCM(nn.Module):
    def __init__(self, in_channels, kernel_size, num_filters):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size, padding=kernel_size // 2, groups=in_channels)
        self.conv3 = nn.Conv1d(in_channels, num_filters, kernel_size, padding=kernel_size // 2)
        self.relu = nn.ReLU()

    def forward(self, x):  # [B, C, T]
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))  # [B, F, T]
        return x

# === Regional Transformer ===
class RTM(nn.Module):
    def __init__(self, embed_dim, heads, blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, heads) for _ in range(blocks)])

    def forward(self, x):  # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.blocks(x)
        return x.transpose(1, 2)  # [B, C, T]

# === Synchronous Transformer ===
class STM(nn.Module):
    def __init__(self, embed_dim, heads, blocks):
        super().__init__()
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, heads) for _ in range(blocks)])

    def forward(self, x):  # [B, C, T]
        x = x.transpose(1, 2)  # [B, T, C]
        x = self.blocks(x)
        return x.transpose(1, 2)

# === Temporal Transformer ===
class TTM(nn.Module):
    def __init__(self, num_segments, embed_dim, heads, blocks):
        super().__init__()
        self.num_segments = num_segments
        self.blocks = nn.Sequential(*[TransformerBlock(embed_dim, heads) for _ in range(blocks)])
        self.ln = nn.LayerNorm(embed_dim)

    def forward(self, x):  # [B, C, T]
        B, C, T = x.shape
        seg_len = T // self.num_segments
        segments = []

        for i in range(self.num_segments):
            seg = x[:, :, i * seg_len:(i + 1) * seg_len]  # [B, C, seg_len]
            seg_mean = seg.mean(dim=2)  # [B, C]
            segments.append(seg_mean.unsqueeze(1))  # [B, 1, C]

        x = torch.cat(segments, dim=1)  # [B, M, C]
        x = self.blocks(x)
        return self.ln(x)  # [B, M, C]

# === CNN Decoder ===
class CNNDecoder(nn.Module):
    def __init__(self, input_dim, num_classes):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv1d(input_dim, 64, kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=1),
            nn.ReLU(),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):  # [B, M, C]
        x = x.transpose(1, 2)         # [B, C, M]
        x = self.decoder(x)           # [B, 64, M]
        x = x.mean(dim=2)             # [B, 64]
        return self.classifier(x)     # [B, num_classes]

# === EEGformer ===
class EEGformer(nn.Module):
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
        x = self.odcm(x)   # [B, 120, T]
        x = self.rtm(x)
        x = self.stm(x)
        x = self.ttm(x)    # [B, M, 120]
        return self.decoder(x)
