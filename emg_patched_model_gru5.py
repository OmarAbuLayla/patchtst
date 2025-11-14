from __future__ import annotations
from typing import List
import torch
import torch.nn as nn


# ============================================================
# Channel Projector (no LayerNorm to preserve EMG amplitude info)
# ============================================================
class ChannelProjector(nn.Module):
    """Projects one EMG channel's patch into embedding space without normalizing away amplitude differences."""

    def __init__(self, patch_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(patch_dim, proj_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# EMG PatchTST + GRU Hybrid (fixed architecture)
# ============================================================
class EMG_GRU_PatchTST(nn.Module):
    """PatchTST-style encoder with residual channel mixing, Transformer refinement, and GRU classifier."""

    def __init__(
        self,
        input_dim: int = 768,
        *,
        num_channels: int = 6,
        proj_dim: int = 96,
        hidden_dim: int = 256,
        num_layers: int = 1,
        num_classes: int = 101,
        dropout: float = 0.12,
        bidirectional: bool = True,
        transformer_layers: int = 2,
        transformer_heads: int = 4,
        transformer_dropout: float = 0.1,
        max_seq_len: int = 512,
    ) -> None:
        super().__init__()
        if input_dim % num_channels != 0:
            raise ValueError("input_dim must be divisible by num_channels")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.input_dim = input_dim
        self.num_channels = num_channels
        self.channel_patch_dim = input_dim // num_channels
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.max_seq_len = max_seq_len

        # Per-channel linear projections (no LayerNorm)
        self.channel_projectors = nn.ModuleList(
            ChannelProjector(self.channel_patch_dim, proj_dim)
            for _ in range(num_channels)
        )

        # Residual channel mixer (preserves inter-channel information)
        self.channel_mixer = nn.Sequential(
            nn.Linear(num_channels * proj_dim, num_channels * proj_dim),
            nn.GELU(),
            nn.Linear(num_channels * proj_dim, proj_dim),
        )

        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer encoder (for local temporal modeling)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=proj_dim,
            nhead=transformer_heads,
            dim_feedforward=proj_dim * 4,
            dropout=transformer_dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=transformer_layers
        )
        self.transformer_dropout = nn.Dropout(transformer_dropout)

        # GRU head for sequential decoding
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        head_input_dim = hidden_dim * (2 if bidirectional else 1)
        self.head = nn.Sequential(
            nn.LayerNorm(head_input_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(head_input_dim, num_classes),
        )

        # Positional embeddings
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_seq_len, proj_dim))
        self.register_buffer(
            "last_channel_embedding_norms", torch.zeros(num_channels), persistent=False
        )

        self._init_weights()

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------
    def _init_weights(self) -> None:
        for projector in self.channel_projectors:
            for module in projector.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        for module in self.channel_mixer:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

        nn.init.trunc_normal_(self.pos_embedding, std=0.02)

        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

        for module in self.head:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    # --------------------------------------------------------
    # Positional encoding
    # --------------------------------------------------------
    def _apply_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        seq_len = tokens.size(1)
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"Sequence length {seq_len} exceeds max_seq_len={self.max_seq_len}"
            )
        return tokens + self.pos_embedding[:, :seq_len, :]

    # --------------------------------------------------------
    # Forward
    # --------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}, got {feature_dim}"
            )

        # (B, T, Channels, PatchLen)
        x = x.view(batch, seq_len, self.num_channels, self.channel_patch_dim)

        channel_embeddings: List[torch.Tensor] = []
        channel_norms = []

        for channel_idx, projector in enumerate(self.channel_projectors):
            channel_patch = x[:, :, channel_idx, :]
            channel_patch = channel_patch.reshape(batch * seq_len, self.channel_patch_dim)
            projected = projector(channel_patch)
            projected = projected.view(batch, seq_len, self.proj_dim)
            channel_embeddings.append(projected)
            channel_norms.append(projected.norm(dim=-1).mean())

        channel_embeddings_tensor = torch.stack(channel_embeddings, dim=2)
        if channel_norms:
            norms_tensor = torch.stack(channel_norms)
            self.last_channel_embedding_norms = norms_tensor.detach()

        # Residual channel mixing
        mixed_flat = channel_embeddings_tensor.reshape(batch * seq_len, -1)
        mixed_residual = mixed_flat
        mixed = self.channel_mixer(mixed_flat) + 0.1 * mixed_residual.mean(dim=1, keepdim=True)
        mixed = mixed.view(batch, seq_len, self.proj_dim)

        mixed = self.embedding_dropout(mixed)
        mixed = self._apply_positional_encoding(mixed)
        mixed = self.transformer_dropout(self.transformer(mixed))

        _, hidden = self.gru(mixed)
        num_directions = 2 if self.bidirectional else 1
        hidden = hidden.view(self.num_layers, num_directions, batch, self.hidden_dim)
        last_layer_hidden = hidden[-1]
        final_state = last_layer_hidden.transpose(0, 1).contiguous().view(batch, -1)

        logits = self.head(final_state)
        return logits
