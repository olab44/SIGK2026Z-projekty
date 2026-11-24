"""Dataset returning (cond, rgb, mask) for training.

No augmentation: any flip or shift of the image would change the correct answer for the same
parameters (the position of the sphere in the image is information the network must learn).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from neural.encoding import encode, cond_dim


def make_coord_grid(h: int, w: int) -> torch.Tensor:
    """CoordConv grid (2, h, w): channel 0 = x, channel 1 = y, in NDC [-1, 1] (pixel centres).

    Same convention as u, v in encoding.py: x grows to the right, y grows UPWARDS in the image
    (image row 0 has y ~ +1)."""
    xs = (torch.arange(w, dtype=torch.float32) + 0.5) / w * 2 - 1
    ys = 1 - (torch.arange(h, dtype=torch.float32) + 0.5) / h * 2
    gy, gx = torch.meshgrid(ys, xs, indexing="ij")
    return torch.stack([gx, gy], dim=0)


def load_split(data_dir: Path, split: str) -> list[int]:
    s = json.loads((Path(data_dir) / "split.json").read_text())
    return list(s[split])


class PhongDataset(Dataset):
    """Returns (cond[D], rgb[3,128,128] in [0,1], mask[1,128,128]) for the selected split."""

    def __init__(self, data_dir, split: str = "train", encoding: str = "relative", cache: bool = True):
        self.data_dir = Path(data_dir)
        self.encoding = encoding
        self.frames = load_split(self.data_dir, split)
        df = pd.read_csv(self.data_dir / "params.csv").set_index("frame")
        self.params = df.loc[self.frames]
        self.conds = np.stack([encode(r, encoding) for _, r in self.params.iterrows()])
        self._cache = {} if cache else None  # optional in-RAM cache of the decoded images

    def __len__(self):
        return len(self.frames)

    def load_rgba(self, frame: int) -> np.ndarray:
        return np.array(Image.open(self.data_dir / "images" / f"image_{frame:04}.png").convert("RGBA"))

    def _get_image(self, i):
        if self._cache is not None and i in self._cache:
            return self._cache[i]
        rgba = self.load_rgba(self.frames[i])
        mask = rgba[..., 3] > 127
        rgb = rgba[..., :3].astype(np.float32) / 255.0 * mask[..., None]   # black background
        out = (torch.from_numpy(rgb).permute(2, 0, 1).contiguous(),
               torch.from_numpy(mask.astype(np.float32))[None])
        if self._cache is not None:
            self._cache[i] = out
        return out

    def __getitem__(self, i):
        rgb, mask = self._get_image(i)
        return torch.from_numpy(self.conds[i]), rgb, mask


class SyntheticDataset(Dataset):
    """Artificial data for smoke tests (when no real data exists): bright discs on a black background."""

    def __init__(self, n: int = 16, encoding: str = "relative", size: int = 128, seed: int = 0):
        rng = np.random.default_rng(seed)
        d = cond_dim(encoding)
        self.conds = rng.uniform(-1, 1, size=(n, d)).astype(np.float32)
        ys, xs = np.mgrid[0:size, 0:size]
        self.rgb, self.mask = [], []
        for _ in range(n):
            cx, cy, r = rng.uniform(20, size - 20), rng.uniform(20, size - 20), rng.uniform(4, 12)
            m = ((xs - cx) ** 2 + (ys - cy) ** 2 <= r * r).astype(np.float32)
            col = rng.uniform(0.2, 1.0, size=3).astype(np.float32)
            self.rgb.append(torch.from_numpy(col[:, None, None] * m[None]))
            self.mask.append(torch.from_numpy(m[None]))

    def __len__(self):
        return len(self.conds)

    def __getitem__(self, i):
        return torch.from_numpy(self.conds[i]), self.rgb[i], self.mask[i]
