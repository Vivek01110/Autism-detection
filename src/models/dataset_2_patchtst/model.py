"""
PatchTST (Patch Time Series Transformer) for Dataset 2 Eye-Tracking Sequence Classification.

Implementation Details:
-----------------------
- Architecture: Pure PyTorch implementation of PatchTST
  (Nie et al., 2023: "A Time Series is Worth 64 Words: Long-term Forecasting with PatchTST").
- Input shape: (B, 200, 8) where B is batch size, 200 is sequence length, and 8 is eye-tracking channels.
- Patch length (patch_len): Configurable (default 16).
- Stride: Configurable (default 8).
- Number of patches: (seq_len - patch_len) // stride + 1 = (200 - 16) // 8 + 1 = 24 non-overlapping/sliding patches.
- Channel independence: Each of the 8 channels is patched and projected independently into the embedding dimension.
- Embedding dimension (d_model): Configurable (default 64).
- Positional encoding: Learnable 1D additive positional embeddings of shape (1, num_patches, d_model).
- Number of Transformer layers (n_layers): Configurable (default 2).
- Number of attention heads (n_heads): Configurable (default 4).
- Feedforward dimension (d_ff): Configurable (default 128).
- Dropout: Configurable (default 0.1).
- Pooling strategy: Mean pooling across the patch dimension (num_patches -> 1) per channel,
  yielding (B, 8, d_model) which is flattened to (B, 8 * d_model = 512).
- Classification head: LayerNorm(512) -> Linear(512, 32) -> GELU() -> Dropout(0.1) -> Linear(32, 1) -> Sigmoid().
- Output: ASD probability in [0, 1].
"""

import math
import torch
import torch.nn as nn
from typing import Optional


class PatchTSTClassifier(nn.Module):
    """
    Patch Time Series Transformer for Eye-Tracking Binary ASD Classification.
    """

    def __init__(
        self,
        input_dim: int = 8,
        seq_len: int = 200,
        patch_len: int = 16,
        stride: int = 8,
        d_model: int = 64,
        n_heads: int = 4,
        d_ff: int = 128,
        n_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.patch_len = patch_len
        self.stride = stride
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_ff = d_ff
        self.n_layers = n_layers

        # Compute number of patches
        if seq_len < patch_len:
            raise ValueError(f"Sequence length ({seq_len}) must be >= patch length ({patch_len})")
        self.num_patches = (seq_len - patch_len) // stride + 1

        # Patch projection layer (patch_len -> d_model)
        self.patch_proj = nn.Linear(patch_len, d_model)

        # Learnable additive positional embeddings
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.norm = nn.LayerNorm(d_model)

        # Classification Head: input features = input_dim * d_model
        head_in = input_dim * d_model
        self.head = nn.Sequential(
            nn.LayerNorm(head_in),
            nn.Linear(head_in, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Eye-tracking sequence tensor of shape (B, 200, input_dim)
        Returns:
            ASD probability tensor of shape (B,) with values in [0, 1]
        """
        B, L, C = x.shape

        # Permute to (B, C, L) for 1D sliding patch extraction
        x_c = x.permute(0, 2, 1)

        # Unfold along time dimension into patches: shape (B, C, num_patches, patch_len)
        patches = x_c.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        num_patches = patches.shape[2]

        # Channel independence: reshape into (B * C, num_patches, patch_len)
        patches_flat = patches.reshape(B * C, num_patches, self.patch_len)

        # Linear embedding + Positional encoding
        emb = self.patch_proj(patches_flat) + self.pos_embed[:, :num_patches, :]

        # Pass through Transformer encoder layers
        h = self.transformer(emb)
        h = self.norm(h)

        # Temporal patch mean pooling: (B * C, num_patches, d_model) -> (B * C, d_model)
        h_pool = h.mean(dim=1)

        # Reshape across channels: (B, C * d_model)
        h_flat = h_pool.reshape(B, C * self.d_model)

        # Classification head -> Sigmoid probability
        prob = self.head(h_flat).squeeze(-1)
        return prob
