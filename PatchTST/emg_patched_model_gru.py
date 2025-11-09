# ==============================================================
#  EMG GRU Model for Patched EMG Sequences (PatchTST baseline)
#  Author: Omar A. Layla
# ==============================================================
import torch
import torch.nn as nn


class TemporalAttention(nn.Module):
    """Learnable attention pooling that highlights salient patches."""

    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Soft attention over the time dimension lets the model emphasise
        # informative patches instead of collapsing everything via a flat mean.
        weights = torch.softmax(self.score(x), dim=1)
        return torch.sum(weights * x, dim=1)


class EMG_GRU_PatchTST(nn.Module):
    """
    GRU model for EMG-only classification using pre-patched time-domain inputs.
    Each sample = sequence of N patches (e.g., 366) × patch_length (e.g., 64).
    """

    def __init__(
        self,
        input_dim: int = 64,       # features per patch (patch_len)
        hidden_dim: int = 256,     # GRU hidden size
        num_layers: int = 2,       # number of GRU layers
        num_classes: int = 101,    # number of target classes
        dropout: float = 0.3,      # dropout rate
        bidirectional: bool = True
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = TemporalAttention(out_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Linear(out_dim, num_classes)
        )

        self._init_weights()

    # ----------------------------------------------------------
    def _init_weights(self) -> None:
        for name, param in self.gru.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                # Restore PyTorch's stable orthogonal recurrent init so that the
                # GRU retains long-range dynamics instead of drifting during
                # optimisation.
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)
        nn.init.xavier_uniform_(self.attention.score.weight)
        if self.attention.score.bias is not None:
            nn.init.zeros_(self.attention.score.bias)
        for m in self.classifier:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ----------------------------------------------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, input_dim)
        Returns:
            logits: (B, num_classes)
        """
        out, _ = self.gru(x)              # (B, seq_len, hidden*dir)
        out = self.dropout(out)
        out = self.attention(out)         # attention pooling over time
        logits = self.classifier(out)     # (B, num_classes)
        return logits
