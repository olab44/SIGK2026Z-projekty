"""Figures: reference vs. network grids (best / median / worst by FLIP), loss curves, metric histograms."""
from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from neural.metrics import DEFAULT_CANNY, METRIC_NAMES, flip_error

COLUMNS = ["reference", "network", "|difference|", "FLIP map", "Canny ref", "Canny network"]


def comparison_row(ref, gen, canny_t=DEFAULT_CANNY):
    """Six panels: reference | generated | |diff| | FLIP map | Canny ref | Canny generated."""
    _, fmap = flip_error(ref, gen)
    diff = np.abs(ref.astype(np.int16) - gen.astype(np.int16)).astype(np.uint8)
    flip_rgb = (matplotlib.colormaps["magma"](np.clip(fmap, 0, 1))[..., :3] * 255).astype(np.uint8)
    edges = [cv2.Canny(cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), *canny_t) for x in (ref, gen)]
    return [ref, gen, diff, flip_rgb, *edges]


def crop_window(img, half: int):
    """(y0, y1, x0, x1) of a 2*half window centred on the non-black pixels (spheres are only a few px)."""
    ys, xs = np.nonzero(img.sum(axis=2) > 0)
    h, w = img.shape[:2]
    if len(ys) == 0:
        return 0, h, 0, w
    y0 = int(np.clip(int(ys.mean()) - half, 0, max(h - 2 * half, 0)))
    x0 = int(np.clip(int(xs.mean()) - half, 0, max(w - 2 * half, 0)))
    return y0, min(h, y0 + 2 * half), x0, min(w, x0 + 2 * half)


def plot_cases(pairs, titles, path: Path, zoom: int = 0, canny_t=DEFAULT_CANNY):
    """pairs: list of (ref, gen). zoom > 0 crops around the sphere."""
    fig, axes = plt.subplots(len(pairs), 6, figsize=(13, 2.2 * len(pairs)), squeeze=False)
    for r, ((ref, gen), title) in enumerate(zip(pairs, titles)):
        win = crop_window(ref, zoom) if zoom else (0, ref.shape[0], 0, ref.shape[1])
        for c, panel in enumerate(comparison_row(ref, gen, canny_t)):
            ax = axes[r, c]
            ax.imshow(panel[win[0]:win[1], win[2]:win[3]], cmap="gray" if panel.ndim == 2 else None,
                      interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(COLUMNS[c], fontsize=9)
        axes[r, 0].set_ylabel(title, fontsize=8)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def select_cases(per_image: pd.DataFrame, k: int = 3) -> dict:
    """Frame ids of the best, median and worst k images by FLIP."""
    s = per_image.sort_values("flip").reset_index(drop=True)
    mid = len(s) // 2
    return {"best": s.frame[:k].tolist(), "median": s.frame[mid - k // 2: mid - k // 2 + k].tolist(),
            "worst": s.frame[-k:].tolist()}


def plot_loss_curves(run_dir: Path, path: Path):
    df = pd.read_csv(run_dir / "losses.csv")
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.8))
    axes[0].plot(df.epoch, df.g_l1, label="g_l1 (weighted)")
    axes[0].plot(df.epoch, df.g_mask, label="g_mask (BCE)")
    axes[0].set_title("Generator losses (train)")
    axes[0].legend()
    axes[1].plot(df.epoch, df.d_loss, label="d_loss (LSGAN)")
    axes[1].plot(df.epoch, df.g_adv, label="g_adv")
    axes[1].set_title("Adversarial losses")
    axes[1].legend()
    v = df.dropna(subset=["val_l1"])
    axes[2].plot(v.epoch, v.val_l1, "o-", label="val masked L1")
    twin = axes[2].twinx()
    twin.plot(v.epoch, v.val_ssim, "s--", color="tab:orange", label="val SSIM")
    axes[2].set_title("Validation")
    axes[2].legend(loc="upper left")
    twin.legend(loc="upper right")
    for a in axes:
        a.set_xlabel("epoch")
        a.grid(alpha=0.3)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_metric_hists(per_image: pd.DataFrame, path: Path):
    fig, axes = plt.subplots(1, 4, figsize=(15, 3.4))
    for ax, m in zip(axes, METRIC_NAMES):
        ax.hist(per_image[m], bins=30)
        ax.axvline(per_image[m].median(), color="k", ls="--", lw=1, label="median")
        ax.set_title(m.upper())
        ax.legend()
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)
