"""Model definition for the 15-channel EMG recogniser."""

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn


__all__ = ["EMGNet15"]


def _conv_block(in_channels: int, out_channels: int, *, kernel_size: Tuple[int, int] = (3, 3),
                stride: Tuple[int, int] = (1, 1), padding: Tuple[int, int] = (1, 1),
                dropout: float = 0.0) -> nn.Sequential:
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    if dropout > 0:
        layers.append(nn.Dropout2d(p=dropout))
    return nn.Sequential(*layers)


class EMGNet15(nn.Module):
    """CNN + GRU architecture tailored for 15-channel inputs."""

    def __init__(
        self,
        num_classes: int,
        *,
        in_channels: int = 15,
        proj_dim: int = 256,
        gru_hidden: int = 512,
        gru_layers: int = 2,
        every_frame: bool = True,
        gru_dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.every_frame = every_frame

        self.frontend = nn.Sequential(
            _conv_block(in_channels, 64),
            _conv_block(64, 64),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 36x36 -> 18x18
            _conv_block(64, 128, dropout=0.1),
            _conv_block(128, 128),
            nn.MaxPool2d(kernel_size=(2, 2)),  # 18x18 -> 9x9
            _conv_block(128, 256, dropout=0.2),
            _conv_block(256, 256),
            nn.MaxPool2d(kernel_size=(2, 1)),  # reduce frequency, preserve time
            nn.Dropout2d(p=0.3),
        )

        self.temporal_proj = nn.Linear(256, proj_dim)
        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=gru_dropout if gru_layers > 1 else 0.0,
            bidirectional=True,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(gru_hidden * 2),
            nn.Linear(gru_hidden * 2, num_classes),
        )

        self._reset_parameters()

    # ------------------------------------------------------------------
    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------------------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # (B, C, F, T)
        x = self.frontend(x)
        # Average over frequency axis; keep temporal resolution from last pooling
        x = x.mean(dim=2)  # (B, channels, time)
        x = x.transpose(1, 2)  # (B, time, channels)
        x = self.temporal_proj(x)
        gru_out, _ = self.gru(x)
        if self.every_frame:
            logits = self.classifier(gru_out)
        else:
            logits = self.classifier(gru_out[:, -1, :])
        return logits