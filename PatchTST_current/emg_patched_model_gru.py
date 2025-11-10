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
        hidden_dim: int = 192,     # default width tuned for 6×128 patches
        num_layers: int = 1,       # default to a single layer to curb overfitting
        num_classes: int = 101,    # number of target classes
        dropout: float = 0.3,      # lighter dropout after diagnostics
        bidirectional: bool = True,
        proj_dim: int = 256,       # bottleneck size before the GRU
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional

        # Project high-dimensional patches (e.g., 6×128) into a compact embedding
        # before the recurrent stack.  This mirrors the diagnostic recommendation
        # to add a bottleneck that improves conditioning and combats overfitting.
        self.input_proj = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, proj_dim),
            nn.GELU(),
            nn.Dropout(p=0.25),  # light dropout before the GRU to regularise tokens
        )

        self.gru = nn.GRU(
            input_size=proj_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )

        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = TemporalAttention(out_dim)
        self.sequence_dropout = nn.Dropout(dropout)
        self.head_dropout = nn.Dropout(0.2)
        self.classifier = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(0.2),
            nn.Linear(out_dim, num_classes),
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
        for module in self.input_proj:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
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
        x = self.input_proj(x)
        out, _ = self.gru(x)              # (B, seq_len, hidden*dir)
        out = self.sequence_dropout(out)
        out = self.attention(out)         # attention pooling over time
        out = self.head_dropout(out)

        logits = self.classifier(out)     # (B, num_classes)
        return logits
