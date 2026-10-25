"""Shared SIGK restoration architecture for denoising and deblurring.

Copy this file unchanged between the two projects. Both tasks instantiate
ResidualUNet() and train separate weights. No project-specific imports or
additional dependencies beyond PyTorch are needed.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


MODEL_ARCHITECTURE = "residual_unet_v1"
MODEL_CHANNELS = (16, 32, 64, 128)


class DoubleConv(nn.Module):
    """Two size-preserving convolutions, each followed by ReLU."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.layers(image)


class UpBlock(nn.Module):
    """Bilinear upsample, reduce channels, concatenate skip, double convolution."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=True),
            nn.ReLU(inplace=True),
        )
        self.fuse = DoubleConv(2 * out_channels, out_channels)

    def forward(self, image: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        image = F.interpolate(image, scale_factor=2, mode="bilinear", align_corners=False)
        image = self.reduce(image)
        return self.fuse(torch.cat((skip, image), dim=1))


class ResidualUNet(nn.Module):
    """Fixed RGB residual U-Net: channels 16/32/64/128, three down/up stages.

    Input: B x 3 x H x W, with H and W divisible by eight (the project uses
    256 x 256). Output has the same shape. The correction is clean - degraded,
    so forward returns degraded + correction for both tasks. There is no batch
    normalization, dropout, output activation, or clamping inside the model.
    The zero-initialized 1 x 1 head makes the initial mapping exactly identity.
    """

    def __init__(self) -> None:
        super().__init__()
        self.encoder1 = DoubleConv(3, 16)
        self.encoder2 = DoubleConv(16, 32)
        self.encoder3 = DoubleConv(32, 64)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bottleneck = DoubleConv(64, 128)
        self.decoder3 = UpBlock(128, 64)
        self.decoder2 = UpBlock(64, 32)
        self.decoder1 = UpBlock(32, 16)
        self.head = nn.Conv2d(16, 3, kernel_size=1, bias=True)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, degraded: torch.Tensor) -> torch.Tensor:
        if degraded.ndim != 4 or degraded.shape[1] != 3:
            raise ValueError("ResidualUNet expects B x 3 x H x W RGB tensors.")
        if degraded.shape[-2] % 8 or degraded.shape[-1] % 8:
            raise ValueError("ResidualUNet image height and width must be divisible by 8.")
        skip1 = self.encoder1(degraded)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        features = self.bottleneck(self.pool(skip3))
        features = self.decoder3(features, skip3)
        features = self.decoder2(features, skip2)
        features = self.decoder1(features, skip1)
        return degraded + self.head(features)
