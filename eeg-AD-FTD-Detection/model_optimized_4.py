# models.py
import torch
import torch.nn as nn


# === Depthwise Temporal Convolution ===
class ODCM(nn.Module):
    def __init__(self, in_channels, kernel_size, num_filters):
        super().__init__()
        S = in_channels
        C = num_filters # 120
        self.conv1 = nn.Conv1d(in_channels=S, out_channels=S*C, kernel_size=kernel_size, padding=0, groups=S, bias=False) # kernel_size = 10
        self.bn1 = nn.BatchNorm1d(S*C)
        
        self.conv2 = nn.Conv1d(in_channels=S*C, out_channels=S*C, kernel_size=kernel_size, padding=0, groups=S*C, bias=False)
        self.bn2 = nn.BatchNorm1d(S*C)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.2)

    def forward(self, x):  # x = [B = Batch, S = Channels, L = Time] , C = Filters
        B,S,L = x.shape
        # 1st conv block
        z1 = self.conv1(x)
        z1 = self.bn1(z1)
        z1 = self.relu(z1)
        z1 = self.dropout(z1) # z1 = [B, S*C, L1]
        
        z2 = self.conv2(z1)
        z2 = self.bn2(z2)
        z2 = self.relu(z2)
        z2 = self.dropout(z2) # z2 = [B, S*C*C, L2]
        
        # 4D reshape
        _,SC,L3 = z2.shape
        C = SC//S
        z3 = z2.view(B,S,C,L3)

        return z3 # [B, S, C, L3]

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dropout):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = nn.MultiheadAttention(embed_dim=dim, num_heads=heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.attn_dropout = nn.Dropout(dropout)
        # Feedforward MLP (dim->4*dim->dim)
        self.mlp   = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # 1) Self-Attention: Q=K=V=LN(x), attn_output: [B, Seq, D]
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x))
        y = self.attn_dropout(y)
        # 2) Residual connection
        x = x + y
        # 3) Position-wise MLP + Residual
        x = x + self.mlp(self.norm2(x))
        return x

"""
- Input z3: [B, S, C, L_e]
- 공간 축(S)을 따라 S개의 sub-matrix [C, L_e] 생성
- 각 sub-matrix의 C개 row를 토큰(token)으로 취급 → 총 S*C tokens
- 각 토큰은 R^{L_e} → R^{D} 로 선형 투영 → Transformer에 입력
"""

# === Regional Transformer Module ===
class RTM(nn.Module):
    def __init__(self,
                 num_regions:int,   # = S (# of Channels)
                 num_filters:int,   # = C (ODCM num_filters)
                 seq_len:int,       # = L_e (ODCM output time length)
                 embed_dim:int,     # Embedding Dimension D (D==C)
                 num_heads:int,
                 num_blocks:int):
        super().__init__()
        S, C, L, D = num_regions, num_filters, seq_len, embed_dim
        self.S, self.C, self.L = S, C, L
        
        self.token_dropout = nn.Dropout(p=0.2)

        # 1) linear patch embedding: R^{L_e} → R^{D}
        self.token_embed = nn.Linear(L, D)

        # 2) Classification Token
        self.cls_token   = nn.Parameter(torch.randn(1, 1, D))

        # 3) Positional Embedding (Combined Sequence length = 1 + S*C)
        self.pos_embed   = nn.Parameter(torch.randn(1, 1 + S*C, D))

        # 4) Transformer Blocks
        self.blocks = nn.Sequential(*[TransformerBlock(D, num_heads, dropout=0.2) for _ in range(num_blocks)])

    def forward(self, x):
        """
        x: [B, S, C, L]  
        returns: [B, S, C, D]
        """
        B, S, C, L = x.shape
        assert S==self.S and C==self.C and L==self.L

        # 1) Convert into token: [B, S, C, L] → [B, S*C, L]
        tokens = x.reshape(B, S*C, L)

        # 2) Linear Projection Embedding: [B, S*C, L] → [B, S*C, D]
        tokens = self.token_embed(tokens)
        tokens = self.token_dropout(tokens)

        # 3) Add classification token: → [B, 1+S*C, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)
        seq = torch.cat((cls_tokens, tokens), dim=1)

        # 4) Add positional embedding
        seq = seq + self.pos_embed

        # 5) Transformer Blocks
        seq = self.blocks(seq)  # [B, 1+S*C, D]

        # 6) Remove classification token → [B, S*C, D]
        seq = seq[:, 1:, :]

        # 7) return back to 4D → [B, S, C, D]
        out = seq.reshape(B, S, C, -1)
        return out


"""
입력 z4 ∈ R^{S×C×D} 에서
- C feature map별 submatrix X_syn_i ∈ R^{S×D}
- 각 row(D) → token (총 C*S tokens)
- Linear: D→D, pos_emb 추가, TransformerBlock 적용 :contentReference[oaicite:4]{index=4}&#8203;:contentReference[oaicite:5]{index=5}
출력 z5 ∈ R^{C×S×D}
"""
# === Synchronous Transformer Module ===
class STM(nn.Module):
    def __init__(self,
                 num_regions: int,   # S (# of channels)
                 num_filters: int,   # C (# of depthwise Filter)
                 embed_dim: int,     # D (Embedding Dimension, same as num_filters)
                 num_heads: int,
                 num_blocks: int):
        super().__init__()
        S, C, D = num_regions, num_filters, embed_dim
        self.S, self.C, self.D = S, C, D
        
        self.token_dropout = nn.Dropout(0.2)

        # For each token, linear patch embedding D->D mapping
        self.token_embed = nn.Linear(D, D)
        self.cls_token   = nn.Parameter(torch.randn(1, 1, D))
        self.pos_embed   = nn.Parameter(torch.randn(1, 1 + S*C, D))
        self.blocks = nn.Sequential(*[TransformerBlock(dim=D, heads=num_heads, dropout=0.2) for _ in range(num_blocks)])

    def forward(self, x):
        """
        x: [B, S, C, D]
        returns: [B, S, C, D]
        """
        B, S, C, D = x.shape
        assert (S, C, D) == (self.S, self.C, self.D)

        # 1) [B, S, C, D] → [B, C, S, D]
        x_permuted = x.permute(0, 2, 1, 3)

        # 2) [B, C, S, D] → [B, C*S, D]  (flatten each sub-matrix's row into sequence)
        tokens = x_permuted.reshape(B, C * S, D)

        # 3) Linear Embedding (D→D)
        tokens = self.token_embed(tokens)  # [B, S*C, D]
        tokens = self.token_dropout(tokens)

        # 4) Add classification token
        cls_tokens = self.cls_token.expand(B, -1, -1) # [B, 1, D]
        seq = torch.cat((cls_tokens, tokens), dim=1) # [B, 1+S*C, D]

        # 5) Add Positional Embedding
        seq = seq + self.pos_embed # [B, 1+S*C, D]

        # 6) TransformerBlock × K
        seq = self.blocks(seq) # [B, 1+S*C, D]

        # 7) Remove Classification token
        seq = seq[:, 1:, :] # [B, S*C, D]

        # 8) Return to original 4D tensor: [B, C, S, D] → [B, S, C, D]
        out = seq.reshape(B, C, S, D).permute(0, 2, 1, 3)
        return out

# === Temporal Transformer Module ===
class TTM(nn.Module):
    def __init__(self,
                 num_segments: int,   # M, temporal segments
                 num_regions: int,   # S, # of channels
                 num_filters: int,   # C, # of filters
                 embed_dim: int,   # D, Embedding Dimension (After RTM, STM)
                 num_heads: int,
                 num_blocks: int,
                 dropout: float = 0.2):
        super().__init__()
        self.M = num_segments
        self.S = num_regions
        self.C = num_filters
        self.D = embed_dim
        
        self.token_dropout = nn.Dropout(dropout)

        # 1) Token Embedding: R^{S*C} → R^{D}
        self.token_embed = nn.Linear(self.S * self.C, self.D)

        # 2) classification token
        self.cls_token = nn.Parameter(torch.randn(1, 1, self.D))

        # 3) positional embedding: total tokens = 1 + M
        self.pos_embed = nn.Parameter(torch.randn(1, 1 + self.M, self.D))

        # 4) Transformer blocks
        self.blocks = nn.Sequential(*[TransformerBlock(dim=self.D, heads=num_heads, dropout=dropout) for _ in range(num_blocks)])
        # 5) LayerNorm
        self.ln = nn.LayerNorm(self.D)
        self.out_proj = nn.Linear(self.D, self.S * self.C)

    def forward(self, x):
        """
        x: [B, S, C, D]
        returns: [B, M, D]
        """
        B, S, C, D = x.shape
        assert (S, C, D) == (self.S, self.C, self.D), \
            f"Expected shape (S,C,D)=({self.S},{self.C},{self.D}), got {(S,C,D)}"

        # 1) Temporal (D) → M segments: Average into M submatrixes
        seg_len = D // self.M
        segments = []
        for i in range(self.M):
            seg = x[:, :, :, i*seg_len : (i+1)*seg_len]  # [B, S, C, seg_len]
            seg_mean = seg.mean(dim=-1)                  # [B, S, C]
            segments.append(seg_mean.unsqueeze(1))       # [B, 1, S, C]
        # Concatatenate M submatrixes -> [B, M, S, C]
        temp = torch.cat(segments, dim=1)

        # 2) Token Sequence: [B, M, S, C] → [B, M, S*C]
        tokens = temp.reshape(B, self.M, self.S * self.C)

        # 3) Linear projection: [B, M, S*C] → [B, M, D]
        tokens = self.token_embed(tokens)
        tokens = self.token_dropout(tokens)

        # 4) Classification Token: → [B, 1+M, D]
        cls_tokens = self.cls_token.expand(B, -1, -1)   # [B,1,D]
        seq = torch.cat((cls_tokens, tokens), dim=1)    # [B,1+M,D]

        # 5) Add positional embedding
        seq = seq + self.pos_embed                      # [B,1+M,D]

        # 6) Transformer blocks
        seq = self.blocks(seq)                          # [B,1+M,D]

        # 7) remove CLS token → [B, M, D]
        seq = seq[:, 1:, :]

        seq = self.ln(seq)
        out = self.out_proj(seq) # [B, M, S*C]
        
        return out


# === CNN Decoder ===
class CNNDecoder(nn.Module):
    def __init__(
        self,
        num_regions: int,    # S: number of channels
        num_filters: int,    # C: number of depthwise filters
        num_segments: int,   # M: number of temporal segments after TTM
        num_classes: int     # number of output classes
    ):
        super().__init__()
        self.num_regions = num_regions
        self.num_filters = num_filters
        self.num_segments = num_segments

        # 1) channel-fusion: C → 1 via 1×1 Conv2d
        #    input: [B, C, S, M] → [B, 1, S, M]
        self.conv1 = nn.Conv2d(in_channels=self.num_filters, out_channels=1, kernel_size=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(1)
        self.conv1_dropout = nn.Dropout2d(p=0.2)

        # 2) spatial-fusion: S → N
        #    input: [B, 1, S, M] → [B, N, 1, M]
        N = 16
        self.conv2 = nn.Conv2d(in_channels=1,out_channels=N,kernel_size=(self.num_regions, 1),bias=False)
        self.bn2 = nn.BatchNorm2d(N)
        self.conv2_dropout = nn.Dropout2d(p=0.2)

        # 3) temporal down-sample: M → M//2
        #    input: [B, N, 1, M] → [B, N2, 1, M//2]
        N2 = 32
        self.conv3 = nn.Conv2d(in_channels=N,out_channels=N2,kernel_size=(1, 2),stride=(1, 2),bias=False)
        self.bn3 = nn.BatchNorm2d(N2)
        self.conv3_dropout = nn.Dropout2d(p=0.2)

        # 4) final classifier
        #    flattened features: N2 * (M//2)
        self.fc   = nn.Linear(N2 * (self.num_segments // 2), num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, O: torch.Tensor):
        """
        Args:
            O: Tensor of shape [B, M, S*C]
            M = num_segments, S = num_regions, C = num_filters
        Returns:
            logits: Tensor of shape [B, num_classes]
        """
        B, M, SC = O.shape
        assert (M, SC) == (self.num_segments, self.num_regions * self.num_filters), \
            f"Expected O.shape=(B, {self.num_segments}, {self.num_regions * self.num_filters}), got {O.shape}"

        # reshape back to 4D: [B, M, S, C] → permute to [B, C, S, M]
        x = O.view(B, M, self.num_regions, self.num_filters) \
             .permute(0, 3, 2, 1)  # → [B, C, S, M]

        # 1) channel-fusion
        x = self.relu(self.bn1(self.conv1(x)))   # [B, 1, S, M]
        x = self.conv1_dropout(x)

        # 2) spatial-fusion
        x = self.relu(self.bn2(self.conv2(x)))   # [B, N, 1, M]
        x = self.conv2_dropout(x)
        
        # 3) temporal down-sample
        x = self.relu(self.bn3(self.conv3(x)))   # [B, N2, 1, M//2]
        x = self.conv3_dropout(x)
        
        # 4) flatten and classify
        x = x.view(B, -1)              # [B, N2 * (M//2)]
        x = self.dropout(x)            # Dropout to prevent overfitting
        logits = self.fc(x)            # [B, num_classes]

        return logits


# === EEGformer Pipeline ===
class EEGformer(nn.Module):
    def __init__(self,
                 in_channels: int,   # S
                 input_length: int,  # L
                 kernel_size: int,   # ODCM kernel
                 num_filters: int,   # C
                 num_heads: int,
                 num_blocks: int,
                 num_segments: int,  # M
                 num_classes: int):
        super().__init__()
        self.in_channels  = in_channels
        self.kernel_size  = kernel_size
        self.num_filters  = num_filters
        self.input_length = input_length

        # 1) Depthwise Temporal Conv Module
        self.odcm = ODCM(in_channels=in_channels,kernel_size=kernel_size,num_filters=num_filters)

        # 2)seq_len ->  L_e = L - 2*(kernel_size-1)
        # Decreased Convolutional Layers in ODCM from 3 layers to 2 layers
        seq_len   = input_length - 2 * (kernel_size - 1)
        # D = num_filters
        embed_dim = num_filters

        # 3) Regional Transformer
        self.rtm = RTM(num_regions=in_channels,num_filters=num_filters,seq_len=seq_len,
                       embed_dim=embed_dim,num_heads=num_heads,num_blocks=num_blocks)

        # 4) Synchronous Transformer
        self.stm = STM(num_regions=in_channels,num_filters=num_filters,embed_dim=embed_dim,
                       num_heads=num_heads,num_blocks=num_blocks)

        # 5) Temporal Transformer
        self.ttm = TTM(num_segments=num_segments,num_regions=in_channels,num_filters=num_filters,
                       embed_dim=embed_dim,num_heads=num_heads,num_blocks=num_blocks)

        # 6) CNN Decoder
        self.decoder = CNNDecoder(num_regions=in_channels,num_filters=num_filters,
                                  num_segments=num_segments,num_classes=num_classes)

    def forward(self, x):
        """
        x: [B, S, L]
        """
        # 1) Depthwise conv → [B, S, C, L_e]
        z3 = self.odcm(x)
        # 2) Regional transformer → [B, S, C, D]
        z4 = self.rtm(z3)
        # 3) Synchronous transformer → [B, S, C, D]
        z5 = self.stm(z4)
        # 4) Temporal transformer → [B, M, S*C]
        z6 = self.ttm(z5)
        # 5) Decoder → [B, num_classes]
        logits = self.decoder(z6)
        return logits