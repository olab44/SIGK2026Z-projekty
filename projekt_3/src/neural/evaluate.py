"""Evaluate a trained generator on the test split (600 images) and make the figures.

    python -m neural.evaluate --checkpoint ../runs/relative_gan/best.pt

Writes to <run dir>/eval_<split>/: images/*.png, per_image.csv, results.json, results.md, figures/*.png.
This is the only place (besides inference) where the test split is used - never for training or model selection.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from utils import config as cfg
from neural import visualize as viz
from neural.dataset import PhongDataset
from neural.metrics import DEFAULT_CANNY, METRIC_NAMES, MetricSuite, summarize
from neural.models import load_generator
from neural.train import generate, pick_device, to_uint8


def load_reference_rgb(ds: PhongDataset, i: int) -> np.ndarray:
    return ds.load_rgba(ds.frames[i])[..., :3].copy()


@torch.no_grad()
def generate_images(G, ds: PhongDataset, device, batch_size: int = 50) -> np.ndarray:
    """Generated images (N, H, W, 3) uint8, in the order of ds.frames."""
    out = []
    for s in range(0, len(ds), batch_size):
        cond = torch.from_numpy(ds.conds[s:s + batch_size]).to(device)
        out.append(to_uint8(generate(G, cond)))
    return np.concatenate(out)


def mean_train_image(data_dir: Path) -> np.ndarray:
    ds = PhongDataset(data_dir, "train", "relative", cache=False)
    acc = np.zeros((cfg.IMAGE_SIZE[1], cfg.IMAGE_SIZE[0], 3), np.float64)
    for i in range(len(ds)):
        acc += load_reference_rgb(ds, i)
    return np.round(acc / len(ds)).astype(np.uint8)


def score(suite: MetricSuite, refs, gens, desc: str) -> pd.DataFrame:
    return pd.DataFrame([suite(r, g) for r, g in tqdm(list(zip(refs, gens)), desc=desc, leave=False)])


def summary(df: pd.DataFrame) -> dict:
    out = summarize({m: df[m].to_numpy() for m in METRIC_NAMES})
    out["hausdorff_penalty_count"] = int(df["hausdorff_penalty"].sum())
    return out


def results_markdown(rows: dict) -> str:
    """Table: Method | FLIP | LPIPS | SSIM | Hausdorff (mean ± std)."""
    lines = ["| Method | FLIP ↓ | LPIPS ↓ | SSIM ↑ | Hausdorff ↓ [px] | Hausdorff penalties |", "|---|---|---|---|---|---|"]
    for name, s in rows.items():
        f = lambda m, d=4: f"{s[m]['mean']:.{d}f} ± {s[m]['std']:.{d}f}"  # noqa: E731
        lines.append(f"| {name} | {f('flip')} | {f('lpips')} | {f('ssim')} | {f('hausdorff', 2)} | "
                     f"{s['hausdorff_penalty_count']} |")
    return "\n".join(lines) + "\n"


def run_evaluation(args) -> Path:
    device = pick_device(args.device)
    ck = Path(args.checkpoint)
    out_dir = Path(args.out_dir) if args.out_dir else ck.parent / f"eval_{args.split}"
    (out_dir / "images").mkdir(parents=True, exist_ok=True)

    G, mcfg = load_generator(ck, device)
    ds = PhongDataset(args.data_dir, args.split, mcfg["encoding"], cache=False)
    refs = [load_reference_rgb(ds, i) for i in range(len(ds))]
    gens = generate_images(G, ds, device)
    for f, g in zip(ds.frames, gens):
        Image.fromarray(g, "RGB").save(out_dir / "images" / f"image_{f:04}.png")

    suite = MetricSuite(args.lpips_device, (args.canny_low, args.canny_high))
    df = score(suite, refs, gens, "network")
    df.insert(0, "frame", ds.frames)
    df.to_csv(out_dir / "per_image.csv", index=False)

    # Baselines show how much the black background alone inflates SSIM / LPIPS / FLIP.
    black = np.zeros_like(refs[0])
    rows = {args.method_name: summary(df),
            "black_image": summary(score(suite, refs, [black] * len(refs), "black")),
            "mean_train_image": summary(score(suite, refs, [mean_train_image(args.data_dir)] * len(refs), "mean"))}
    (out_dir / "results.json").write_text(json.dumps(
        {"split": args.split, "n_images": len(ds), "checkpoint": str(ck), "model_config": mcfg,
         "canny": [args.canny_low, args.canny_high], "results": rows}, indent=2))
    table = results_markdown(rows)
    (out_dir / "results.md").write_text(table)
    print(table)

    if not args.no_figures:
        fig_dir = out_dir / "figures"
        pairs, titles = [], []
        for cat, frames in viz.select_cases(df.assign(frame=ds.frames)).items():
            for f in frames:
                i = ds.frames.index(f)
                pairs.append((refs[i], gens[i]))
                titles.append(f"{cat}\n#{f}\nFLIP {df.flip.iloc[i]:.3f}")
        viz.plot_cases(pairs, titles, fig_dir / "cases.png", args.zoom, (args.canny_low, args.canny_high))
        viz.plot_metric_hists(df, fig_dir / "metric_hists.png")
        if (ck.parent / "losses.csv").exists():
            viz.plot_loss_curves(ck.parent, fig_dir / "loss_curves.png")
    print(f"results: {out_dir}")
    return out_dir


def build_parser():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--split", default="test", choices=["test", "val", "train"])
    ap.add_argument("--data_dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--out_dir", default=None)
    ap.add_argument("--method_name", default="neural_renderer")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--lpips_device", default="cpu")
    ap.add_argument("--canny_low", type=float, default=DEFAULT_CANNY[0])
    ap.add_argument("--canny_high", type=float, default=DEFAULT_CANNY[1])
    ap.add_argument("--zoom", type=int, default=24, help="half-size of the crop around the sphere in the figures; 0 = full image")
    ap.add_argument("--no_figures", action="store_true")
    return ap


def main(argv=None):
    run_evaluation(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
