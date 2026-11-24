"""Generator (cond -> image) and conditional PatchGAN discriminator, both with CoordConv."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from neural.dataset import make_coord_grid

EMB_DIM = 256
REL_CLAMP = 4.0   # clamp of the relative maps (x-u)/s - inside the sphere |.| <= 1


class FiLM(nn.Module):
    """Feature modulation: x * (1 + gamma) + beta, where (gamma, beta) come from the embedding e."""

    def __init__(self, emb_dim: int, ch: int):
        super().__init__()
        self.proj = nn.Linear(emb_dim, 2 * ch)
        nn.init.zeros_(self.proj.weight)
        nn.init.zeros_(self.proj.bias)

    def forward(self, x, e):
        gamma, beta = self.proj(e).chunk(2, dim=1)
        return x * (1 + gamma[:, :, None, None]) + beta[:, :, None, None]


class ConvNormFiLM(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm = nn.GroupNorm(min(8, out_ch), out_ch, affine=False)   # no affine parameters (FiLM does that)
        self.film = FiLM(emb_dim, out_ch)

    def forward(self, x, e):
        return F.leaky_relu(self.film(self.norm(self.conv(x)), e), 0.2)


class UpBlock(nn.Module):
    """Upsample x2 (nearest) -> concat CoordConv (+ relative maps) -> (Conv-GN-FiLM-LReLU) x 2."""

    def __init__(self, in_ch, out_ch, emb_dim, use_pixel_rel_maps: bool):
        super().__init__()
        self.use_rel = use_pixel_rel_maps
        extra = 2 + (2 if use_pixel_rel_maps else 0)
        self.first = ConvNormFiLM(in_ch + extra, out_ch, emb_dim)
        self.second = ConvNormFiLM(out_ch, out_ch, emb_dim)

    def forward(self, x, e, uvs):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        b, _, h, w = x.shape
        coord = make_coord_grid(h, w).to(x.device, x.dtype)[None].expand(b, -1, -1, -1)
        feats = [x, coord]
        if self.use_rel:
            u, v, s = uvs[:, 0], uvs[:, 1], uvs[:, 2]
            rx = ((coord[:, 0] - u[:, None, None]) / s[:, None, None]).clamp(-REL_CLAMP, REL_CLAMP)
            ry = ((coord[:, 1] - v[:, None, None]) / s[:, None, None]).clamp(-REL_CLAMP, REL_CLAMP)
            feats += [rx[:, None], ry[:, None]]
        x = self.first(torch.cat(feats, dim=1), e)
        return self.second(x, e)


class ParamToImageGenerator(nn.Module):
    """Scene parameters -> (rgb in [0,1], mask logit). Deterministic, no noise vector z.

    4x4 -> 8 -> 16 -> 32 -> 64 -> 128; channels after the successive blocks: 512, 256, 128, 64, 32
    (for base_ch=512: the first block keeps 512 channels, the following ones halve them).
    With use_pixel_rel_maps=True the cond vector must hold (u, v, s) in columns 0..2,
    i.e. the 'relative' encoding.
    """

    def __init__(self, cond_dim: int, base_ch: int = 512, use_pixel_rel_maps: bool = False):
        super().__init__()
        self.cond_dim, self.base_ch, self.use_pixel_rel_maps = cond_dim, base_ch, use_pixel_rel_maps
        self.mlp = nn.Sequential(nn.Linear(cond_dim, EMB_DIM), nn.SiLU(), nn.Linear(EMB_DIM, EMB_DIM), nn.SiLU())
        self.base_ch = base_ch
        self.fc = nn.Linear(EMB_DIM, base_ch * 4 * 4)
        chs = [base_ch, base_ch // 2, base_ch // 4, base_ch // 8, base_ch // 16]
        self.blocks = nn.ModuleList()
        prev = base_ch
        for c in chs:
            self.blocks.append(UpBlock(prev, c, EMB_DIM, use_pixel_rel_maps))
            prev = c
        self.head = nn.Conv2d(prev, 4, 1)

    def forward(self, cond):
        e = self.mlp(cond)
        x = self.fc(e).view(-1, self.base_ch, 4, 4)
        for blk in self.blocks:
            x = blk(x, e, cond[:, :3])
        out = self.head(x)
        return torch.sigmoid(out[:, :3]), out[:, 3:4]


def compose(rgb: torch.Tensor, mask_logit: torch.Tensor, hard: bool = False):
    """Final image: rgb * mask (soft mask in training, hard mask >0.5 at inference)."""
    prob = torch.sigmoid(mask_logit)
    if hard:
        prob = (prob > 0.5).to(rgb.dtype)
    return rgb * prob


def _sn(conv):
    return nn.utils.spectral_norm(conv)


class CondPatchDiscriminator(nn.Module):
    """PatchGAN (receptive field ~70 px). Input: [RGB (3), cond broadcast over the image (D), coord (2)]."""

    def __init__(self, cond_dim: int):
        super().__init__()
        in_ch = 3 + cond_dim + 2
        self.net = nn.Sequential(
            _sn(nn.Conv2d(in_ch, 64, 4, 2, 1)), nn.LeakyReLU(0.2),            # C64 s2, no normalization
            _sn(nn.Conv2d(64, 128, 4, 2, 1)), nn.LeakyReLU(0.2),              # C128 s2
            _sn(nn.Conv2d(128, 256, 4, 2, 1)), nn.LeakyReLU(0.2),             # C256 s2
            _sn(nn.Conv2d(256, 512, 4, 1, 1)), nn.LeakyReLU(0.2),             # C512 s1
            _sn(nn.Conv2d(512, 1, 4, 1, 1)),                                  # 1-channel patch map
        )

    def forward(self, rgb, cond):
        b, _, h, w = rgb.shape
        coord = make_coord_grid(h, w).to(rgb.device, rgb.dtype)[None].expand(b, -1, -1, -1)
        cond_map = cond[:, :, None, None].expand(-1, -1, h, w)
        return self.net(torch.cat([rgb, cond_map, coord], dim=1))


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


def load_generator(checkpoint, device="cpu"):
    """Load the generator (EMA weights) from best.pt / last.pt. Returns (G in eval mode, config)."""
    ck = torch.load(checkpoint, map_location=device)
    cfg_ = ck["config"]
    G = ParamToImageGenerator(cfg_["cond_dim"], cfg_["base_ch"], cfg_["pixel_rel_maps"]).to(device)
    G.load_state_dict(ck["G_ema"])
    return G.eval(), cfg_
