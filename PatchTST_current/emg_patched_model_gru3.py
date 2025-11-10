"""EMG PatchTST + GRU model with channel-aware patch encoding."""
from __future__ import annotations

from typing import List

import torch
import torch.nn as nn


class ChannelProjector(nn.Module):
    """Project a single channel patch into a lower dimensional embedding."""

    def __init__(self, patch_dim: int, proj_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, proj_dim),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., patch_dim)
        return self.net(x)


class EMG_GRU_PatchTST(nn.Module):
    """PatchTST-style channel-preserving encoder with GRU classifier head."""

    def __init__(
        self,
        input_dim: int = 768,
        *,
        num_channels: int = 6,
        proj_dim: int = 64,
        hidden_dim: int = 192,
        num_layers: int = 1,
        num_classes: int = 101,
        dropout: float = 0.3,
        bidirectional: bool = False,
    ) -> None:
        super().__init__()
        if input_dim % num_channels != 0:
            raise ValueError(
                "input_dim must be divisible by num_channels for per-channel patching"
            )

        self.input_dim = input_dim
        self.num_channels = num_channels
        self.channel_patch_dim = input_dim // num_channels
        self.proj_dim = proj_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.channel_projectors = nn.ModuleList(
            ChannelProjector(self.channel_patch_dim, proj_dim) for _ in range(num_channels)
        )
        self.channel_dropout = nn.Dropout(dropout)

        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim * (2 if bidirectional else 1)),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim * (2 if bidirectional else 1), num_classes),
        )

        self.register_buffer("last_channel_embedding_norms", torch.zeros(num_channels), persistent=False)
        self._init_weights()

    def _init_weights(self) -> None:
        for projector in self.channel_projectors:
            for module in projector.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tensor of shape (batch, seq_len, input_dim)
        Returns:
            Tensor of shape (batch, num_classes)
        """
        batch, seq_len, feature_dim = x.shape
        if feature_dim != self.input_dim:
            raise ValueError(
                f"Expected input feature dimension {self.input_dim}, got {feature_dim}"
            )

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

        aggregated = channel_embeddings_tensor.mean(dim=2)
        aggregated = self.channel_dropout(aggregated)

        _, hidden = self.gru(aggregated)
        num_directions = 2 if self.bidirectional else 1
        hidden = hidden.view(self.num_layers, num_directions, batch, self.hidden_dim)
        last_layer_hidden = hidden[-1]
        final_state = last_layer_hidden.transpose(0, 1).contiguous().view(batch, -1)

        logits = self.head(final_state)
        return logits
