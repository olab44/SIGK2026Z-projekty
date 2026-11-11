"""
Architecture: NAF-U-Net (NAFNet blocks in a 4-level U-Net, width 32) with a
global input -> output skip. Restoration is done in the gamma domain: the dark
8-bit input is first linearised, amplified by the known exposure gain (+3 EV,
x8), clipped and gamma-encoded again. The network then only predicts a residual
correction (denoising + tone / colour adjustment towards the ground truth).
"""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F


MODEL_ARCHITECTURE = "naf_unet_v1"
DEFAULT_GAIN = 8.0  # +3 EV, inverse of the -3 EV used to create the dataset
DEFAULT_GAMMA = 2.2


# --------------------------------------------------------------------------- #
# Gamma-domain pre-processing
# --------------------------------------------------------------------------- #
def amplify_gamma(
    image: torch.Tensor, gain: float = DEFAULT_GAIN, gamma: float = DEFAULT_GAMMA
) -> torch.Tensor:
    """Brighten a gamma-encoded image by a fixed exposure gain.

    gamma -> linear (x ** gamma), multiply by ``gain`` in linear light, clip to
    [0, 1] and go back to the gamma domain (x ** (1 / gamma)). The noise is
    amplified together with the signal, which is what the network removes.
    """
    linear = image.clamp(0.0, 1.0) ** gamma
    linear = (linear * gain).clamp(1e-8, 1.0)
    return linear ** (1.0 / gamma)


# --------------------------------------------------------------------------- #
# NAFNet building blocks
# --------------------------------------------------------------------------- #
class LayerNorm2d(nn.Module):
    """Layer normalisation over the channel dimension of a B x C x H x W tensor."""

    def __init__(self, channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(channels))
        self.bias = nn.Parameter(torch.zeros(channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


class SimpleGate(nn.Module):
    """Split channels in half and multiply the halves (non-linearity of NAFNet)."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class SimplifiedChannelAttention(nn.Module):
    """Global average pooling followed by a 1x1 conv, used as channel weights."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv2d(channels, channels, kernel_size=1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.conv(self.pool(x))


class NAFBlock(nn.Module):
    """Nonlinear-Activation-Free block (Chen et al., 2022).

    Spatial part:  LN -> 1x1 (x2 ch) -> depthwise 3x3 -> SimpleGate -> SCA -> 1x1
    Channel part:  LN -> 1x1 (x2 ch) -> SimpleGate -> 1x1
    Both parts are residual with learnable scales initialised to zero, so every
    block starts as the identity.
    """

    def __init__(self, channels: int, dw_expand: int = 2, ffn_expand: int = 2,
                 dropout: float = 0.0) -> None:
        super().__init__()
        dw_ch = channels * dw_expand
        ffn_ch = channels * ffn_expand

        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_ch, kernel_size=1)
        self.conv2 = nn.Conv2d(dw_ch, dw_ch, kernel_size=3, padding=1, groups=dw_ch)
        self.sg = SimpleGate()
        self.sca = SimplifiedChannelAttention(dw_ch // 2)
        self.conv3 = nn.Conv2d(dw_ch // 2, channels, kernel_size=1)

        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_ch, kernel_size=1)
        self.conv5 = nn.Conv2d(ffn_ch // 2, channels, kernel_size=1)

        self.drop1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.drop2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(self.norm1(x))
        y = self.conv2(y)
        y = self.sg(y)
        y = self.sca(y)
        y = self.drop1(self.conv3(y))
        x = x + y * self.beta

        y = self.conv4(self.norm2(x))
        y = self.sg(y)
        y = self.drop2(self.conv5(y))
        return x + y * self.gamma


# --------------------------------------------------------------------------- #
# NAF-U-Net
# --------------------------------------------------------------------------- #
class NAFUNet(nn.Module):
    """4-level U-Net built from NAFBlocks with a global input -> output skip.

    Channels: width, 2w, 4w, 8w in the encoder levels and 16w in the bottleneck
    (32 / 64 / 128 / 256 / 512 for width=32). Downsampling: 2x2 conv, stride 2.
    Upsampling: 1x1 conv + PixelShuffle(2). Encoder features are added (not
    concatenated) to the decoder features. The output head is zero-initialised,
    so the untrained model returns its input exactly.

    Any H x W is accepted: the input is reflect-padded to a multiple of 16 and
    the output is cropped back. There is no clamping inside ``forward``.
    """

    def __init__(
        self,
        in_channels: int = 3,
        out_channels: int = 3,
        width: int = 32,
        enc_blocks: Sequence[int] = (1, 1, 2, 4),
        middle_blocks: int = 4,
        dec_blocks: Sequence[int] = (1, 1, 1, 1),
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        assert len(enc_blocks) == len(dec_blocks), "encoder/decoder depth mismatch"
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.intro = nn.Conv2d(in_channels, width, kernel_size=3, padding=1)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()
        ch = width
        for n in enc_blocks:
            self.encoders.append(nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(n)]))
            self.downs.append(nn.Conv2d(ch, ch * 2, kernel_size=2, stride=2))
            ch *= 2

        self.middle = nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(middle_blocks)])

        self.ups = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for n in dec_blocks:
            self.ups.append(nn.Sequential(
                nn.Conv2d(ch, ch * 2, kernel_size=1, bias=False),
                nn.PixelShuffle(2),
            ))
            ch //= 2
            self.decoders.append(nn.Sequential(*[NAFBlock(ch, dropout=dropout) for _ in range(n)]))

        self.ending = nn.Conv2d(width, out_channels, kernel_size=3, padding=1)
        nn.init.zeros_(self.ending.weight)
        nn.init.zeros_(self.ending.bias)

        self.pad_multiple = 2 ** len(enc_blocks)

    def _pad(self, x: torch.Tensor) -> tuple[torch.Tensor, int, int]:
        _, _, h, w = x.shape
        m = self.pad_multiple
        ph = (m - h % m) % m
        pw = (m - w % m) % m
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph), mode="reflect")
        return x, h, w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        inp, h, w = self._pad(x)
        feat = self.intro(inp)

        skips = []
        for encoder, down in zip(self.encoders, self.downs):
            feat = encoder(feat)
            skips.append(feat)
            feat = down(feat)

        feat = self.middle(feat)

        for decoder, up, skip in zip(self.decoders, self.ups, reversed(skips)):
            feat = up(feat) + skip
            feat = decoder(feat)

        out = self.ending(feat)
        if self.out_channels == self.in_channels:
            out = out + inp  # global input -> output skip
        return out[:, :, :h, :w]


class LowLightEnhancer(nn.Module):
    """Full pipeline: dark gamma-encoded image -> amplify_gamma -> NAF-U-Net.

    ``gain`` and ``gamma`` are fixed constants (registered as buffers), not
    learnable parameters. Input and output are B x 3 x H x W in [0, 1].
    """

    def __init__(self, net: nn.Module | None = None, gain: float = DEFAULT_GAIN,
                 gamma: float = DEFAULT_GAMMA) -> None:
        super().__init__()
        self.net = net if net is not None else NAFUNet()
        self.register_buffer("gain", torch.tensor(float(gain)), persistent=False)
        self.register_buffer("gamma", torch.tensor(float(gamma)), persistent=False)

    def preprocess(self, dark: torch.Tensor) -> torch.Tensor:
        return amplify_gamma(dark, float(self.gain), float(self.gamma))

    def forward(self, dark: torch.Tensor) -> torch.Tensor:
        return self.net(self.preprocess(dark))


# --------------------------------------------------------------------------- #
# Losses
# --------------------------------------------------------------------------- #
class CharbonnierLoss(nn.Module):
    """Smooth L1-like loss: mean(sqrt((x - y)^2 + eps^2))."""

    def __init__(self, eps: float = 1e-3) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.sqrt((pred - target).pow(2) + self.eps ** 2).mean()


def _gaussian_window(size: int, sigma: float, channels: int,
                     device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    coords = torch.arange(size, device=device, dtype=dtype) - (size - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    window = (g[:, None] @ g[None, :]).expand(channels, 1, size, size)
    return window.contiguous()


def ssim(pred: torch.Tensor, target: torch.Tensor, window_size: int = 11,
         sigma: float = 1.5, data_range: float = 1.0) -> torch.Tensor:
    """Mean SSIM (Gaussian window, per channel) for B x C x H x W tensors."""
    c = pred.shape[1]
    window = _gaussian_window(window_size, sigma, c, pred.device, pred.dtype)
    pad = window_size // 2
    mu_x = F.conv2d(pred, window, padding=pad, groups=c)
    mu_y = F.conv2d(target, window, padding=pad, groups=c)
    sxx = F.conv2d(pred * pred, window, padding=pad, groups=c) - mu_x ** 2
    syy = F.conv2d(target * target, window, padding=pad, groups=c) - mu_y ** 2
    sxy = F.conv2d(pred * target, window, padding=pad, groups=c) - mu_x * mu_y
    c1 = (0.01 * data_range) ** 2
    c2 = (0.03 * data_range) ** 2
    num = (2 * mu_x * mu_y + c1) * (2 * sxy + c2)
    den = (mu_x ** 2 + mu_y ** 2 + c1) * (sxx + syy + c2)
    return (num / den).mean()


class RestorationLoss(nn.Module):
    """Charbonnier + ssim_weight * (1 - SSIM)."""

    def __init__(self, ssim_weight: float = 0.1, eps: float = 1e-3) -> None:
        super().__init__()
        self.charbonnier = CharbonnierLoss(eps)
        self.ssim_weight = ssim_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss = self.charbonnier(pred, target)
        if self.ssim_weight > 0:
            loss = loss + self.ssim_weight * (1.0 - ssim(pred, target))
        return loss


# --------------------------------------------------------------------------- #
# Utilities
# --------------------------------------------------------------------------- #
def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def tiled_inference(model: nn.Module, image: torch.Tensor, tile: int = 512,
                    overlap: int = 32) -> torch.Tensor:
    """Run ``model`` on a large 1 x C x H x W image in overlapping tiles.

    Each tile is processed with ``overlap`` pixels of context on every side;
    only its central part is written to the output, which removes seams.
    """
    _, _, h, w = image.shape
    if h <= tile and w <= tile:
        return model(image)

    step = tile - 2 * overlap
    out: torch.Tensor | None = None
    for y0 in range(0, h, step):
        for x0 in range(0, w, step):
            y1, x1 = min(y0 + step, h), min(x0 + step, w)
            ty0, tx0 = max(y0 - overlap, 0), max(x0 - overlap, 0)
            ty1, tx1 = min(y1 + overlap, h), min(x1 + overlap, w)
            pred = model(image[:, :, ty0:ty1, tx0:tx1])
            if out is None:
                out = pred.new_zeros(image.shape[0], pred.shape[1], h, w)
            out[:, :, y0:y1, x0:x1] = pred[:, :, y0 - ty0:y1 - ty0, x0 - tx0:x1 - tx0]
    return out
