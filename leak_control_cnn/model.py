"""The bespoke convolutional encoder used as the leak-control optimization target.

Custom and small: two convolutional encoder blocks followed by a
global-average-pool + linear readout. It is not a published benchmark network,
so there is no leaderboard whose optimum an LLM could have memorized.
``base_channels`` and ``dropout`` are search dimensions; the depth (2 blocks) and
kernel sizes are fixed (architecture figure: plot_cnn_arch_pnn.py).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ConvEncoderClassifier(nn.Module):
    """1xHxW image -> [Conv(C)->BN->ReLU->Pool] -> [Conv(2C)->BN->ReLU->Pool]
    -> AdaptiveAvgPool(4x4) -> Flatten -> Dropout -> Linear(n_classes)."""

    def __init__(
        self,
        base_channels: int,
        dropout: float,
        *,
        in_channels: int = 1,
        n_classes: int = 10,
    ) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = 2 * c1
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, c1, kernel_size=3, padding=1),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.BatchNorm2d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.pool = nn.AdaptiveAvgPool2d(4)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(float(dropout)),
            nn.Linear(c2 * 4 * 4, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.head(self.pool(self.encoder(x)))


def parameter_count(base_channels: int, dropout: float = 0.0) -> int:
    """Trainable-parameter count for a given width (used by the arch figure)."""

    model = ConvEncoderClassifier(base_channels, dropout)
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


__all__ = ["ConvEncoderClassifier", "parameter_count"]
