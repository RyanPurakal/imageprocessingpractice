"""Compact U-Net for single-channel image denoising; receives a noisy (B,1,H,W) tensor and returns a restored tensor of the same shape. Spatial size must be a multiple of 8 (three pooling stages)."""

from __future__ import annotations

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two consecutive Conv2d+ReLU layers used as the basic building block at each U-Net resolution level."""

    def __init__(self, in_ch: int, out_ch: int) -> None:
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x)


class UNetDenoise1ch(nn.Module):
    """Encoder–bottleneck–decoder U-Net trained with MSE against clean fields; skip connections concatenate encoder feature maps into the decoder at each scale to preserve spatial detail."""

    def __init__(self, base: int = 32) -> None:
        super().__init__()
        self.enc1 = DoubleConv(1, base)
        self.pool1 = nn.MaxPool2d(2)
        self.enc2 = DoubleConv(base, base * 2)
        self.pool2 = nn.MaxPool2d(2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool3 = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.out = nn.Conv2d(base, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        c1 = self.enc1(x)
        c2 = self.enc2(self.pool1(c1))
        c3 = self.enc3(self.pool2(c2))
        b = self.bottleneck(self.pool3(c3))
        u3 = self.up3(b)
        u3 = self.dec3(torch.cat([u3, c3], dim=1))
        u2 = self.up2(u3)
        u2 = self.dec2(torch.cat([u2, c2], dim=1))
        u1 = self.up1(u2)
        u1 = self.dec1(torch.cat([u1, c1], dim=1))
        return self.out(u1)
