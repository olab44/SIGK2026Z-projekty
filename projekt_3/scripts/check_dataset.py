"""Check the generated dataset: file count and size, sphere mask, and agreement with the projection.

Among others this catches Y-axis errors (flipped images): the mask centroid must agree with the
analytic (u, v) computed from the parameters in params.csv.

Usage:  python scripts/check_dataset.py [--data_dir data/phong]
"""
import _bootstrap  # noqa: F401  (sets sys.path)

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

from utils import config as cfg
import scene_sampling as ss


def mask_centroid_ndc(mask: np.ndarray):
    """Mask (H, W) bool centroid -> (u, v) in NDC (u to the right, v up)."""
    h, w = mask.shape
    ys, xs = np.nonzero(mask)
    cx, cy = xs.mean() + 0.5, ys.mean() + 0.5          # pixel centres
    return cx / w * 2 - 1, 1 - cy / h * 2


def check_dataset(data_dir: Path, expected: int = cfg.NUM_IMAGES, strict_border: bool = False) -> dict:
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "params.csv")
    errors, warnings, rows = [], [], []
    files = sorted((data_dir / "images").glob("image_*.png"))
    if len(files) != expected:
        errors.append(f"file count: {len(files)} != {expected}")
    if len(df) != expected:
        errors.append(f"CSV row count: {len(df)} != {expected}")

    w_img, h_img = cfg.IMAGE_SIZE
    for _, r in df.iterrows():
        f = data_dir / "images" / f"image_{int(r.frame):04}.png"
        if not f.exists():
            errors.append(f"missing file {f.name}")
            continue
        img = np.array(Image.open(f))
        if img.shape != (h_img, w_img, 4):
            errors.append(f"{f.name}: shape {img.shape}")
            continue
        mask = img[..., 3] > 127
        obj = np.array([r.obj_x, r.obj_y, r.obj_z])
        u, v = ss.project_center_ndc(obj)
        radius_px = ss.apparent_radius_ndc(obj) * w_img / 2
        problems = []
        if not mask.any():
            problems.append("empty mask")
            rows.append(dict(frame=int(r.frame), radius_px=radius_px, err_px=np.nan, area_px=0, problems=problems))
            errors.append(f"{f.name}: empty mask")
            continue
        if mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            # With VISIBILITY_MARGIN = 0 the sphere may be tangent to the frustum and touch an edge
            # pixel - this is a warning (an error only with --strict_border).
            (problems if strict_border else warnings).append(f"{f.name}: mask touches the image border")
        cu, cv = mask_centroid_ndc(mask)
        err_px = float(np.hypot(cu - u, cv - v) * w_img / 2)
        tol = max(2.0, 0.1 * radius_px)
        if err_px > tol:
            problems.append(f"mask centroid {err_px:.2f}px from the projection (tol {tol:.2f})")
        area = int(mask.sum())
        rows.append(dict(frame=int(r.frame), radius_px=radius_px, err_px=err_px, area_px=area, problems=problems))
        errors.extend(p if p.startswith("image_") else f"{f.name}: {p}" for p in problems)
    return {"errors": errors, "warnings": warnings, "details": pd.DataFrame(rows), "params": df}


def plot_histograms(df: pd.DataFrame, details: pd.DataFrame, out_dir: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    groups = [("obj_x", "obj_y", "obj_z"), ("light_x", "light_y", "light_z"),
              ("kd_r", "kd_g", "kd_b"), ("shininess", "attempts")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for ax, cols in zip(axes.ravel(), groups):
        for c in cols:
            ax.hist(df[c], bins=30, alpha=0.55, label=c)
        ax.legend()
    fig.suptitle("Dataset parameter histograms")
    fig.tight_layout()
    fig.savefig(out_dir / "params_hist.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(details["radius_px"], bins=40)
    ax.set_xlabel("sphere radius [px]")
    ax.set_ylabel("number of images")
    ax.set_title("Sphere size in the 128x128 image")
    fig.tight_layout()
    fig.savefig(out_dir / "sphere_radius_px_hist.png", dpi=130)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--expected", type=int, default=cfg.NUM_IMAGES)
    ap.add_argument("--figures_dir", type=Path, default=cfg.FIGURES_DIR)
    ap.add_argument("--no_plots", action="store_true")
    ap.add_argument("--strict_border", action="store_true", help="touching the image border is an error")
    args = ap.parse_args(argv)

    res = check_dataset(args.data_dir, args.expected, args.strict_border)
    d = res["details"]
    if len(d):
        print(f"sphere radius [px]: min {d.radius_px.min():.1f}, mediana {d.radius_px.median():.1f}, "
              f"max {d.radius_px.max():.1f}; spheres with radius < 3 px: {(d.radius_px < 3).sum()}")
        print(f"mask centroid vs projection error [px]: mean {d.err_px.mean():.3f}, max {d.err_px.max():.3f}")
    if not args.no_plots and len(d):
        plot_histograms(res["params"], d, args.figures_dir)
        print(f"histograms saved to {args.figures_dir}")
    if res["warnings"]:
        print(f"WARNINGS ({len(res['warnings'])}): {', '.join(w.split(':')[0] for w in res['warnings'][:20])}")
    if res["errors"]:
        print(f"ERRORS ({len(res['errors'])}):")
        for e in res["errors"][:50]:
            print("  -", e)
        return 1
    print("OK: the dataset satisfies all checked conditions")
    return 0


if __name__ == "__main__":
    sys.exit(main())
