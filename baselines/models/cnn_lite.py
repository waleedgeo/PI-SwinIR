"""Compact residual CNN baseline for PI-SwinIR emergency revision."""

from __future__ import annotations

import torch
from torch import nn


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class CNNLite(nn.Module):
    def __init__(
        self,
        in_channels: int = 11,
        hidden_channels: int = 64,
        num_blocks: int = 8,
        fabdem_channel_idx: int = 6,
        residual_anchor: bool = True,
    ):
        super().__init__()
        self.fabdem_channel_idx = fabdem_channel_idx
        self.residual_anchor = residual_anchor
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_channels) for _ in range(num_blocks)])
        self.head = nn.Conv2d(hidden_channels, 1, 3, padding=1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.head(self.blocks(self.stem(x)))
        if self.residual_anchor:
            return x[:, self.fabdem_channel_idx:self.fabdem_channel_idx + 1] + residual
        return residual


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = CNNLite()
    x = torch.randn(1, 11, 128, 128)
    y = model(x)
    print(f"parameters={count_parameters(model)}")
    print(f"output_shape={tuple(y.shape)}")
