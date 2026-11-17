"""Run the Project 2 checkpoint comparison and two controlled training experiments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import time
from pathlib import Path

import imageio.v2 as imageio
import lpips
import matplotlib
import numpy as np
import pandas as pd
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models import ExposurePerceptualLoss, LowLightEnhancer, NAFUNet, RestorationLoss, tiled_inference

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LOWER_LR = 3e-4
METRICS = ["PSNR", "SSIM", "LPIPS"]


class ImagePairs:
    """Read the same cached PNG pairs as the reference notebook."""

    def __init__(self, data_dir: Path, cfg: dict):
        degradation = {key: cfg[key] for key in ("ev_shift", "alpha", "sigma", "gamma")}
        cache_id = hashlib.sha256(json.dumps(degradation, sort_keys=True).encode()).hexdigest()[:12]
        self.gt_dir = data_dir / "processed" / "gt"
        self.dark_dir = data_dir / "processed" / cache_id / "input"

    def paths(self, name: str) -> tuple[Path, Path]:
        return self.dark_dir / f"{name}_input.png", self.gt_dir / f"{name}_gt.png"

    def check(self, names: list[str]) -> None:
        for name in names:
            for path in self.paths(name):
                if not path.is_file():
                    raise FileNotFoundError(f"Missing {path}. Run the notebook's preprocessing cells first.")

    def load(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        dark_path, gt_path = self.paths(name)
        return imageio.imread(dark_path), imageio.imread(gt_path)


def to_tensor(image: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(image)).permute(2, 0, 1).float().div(255.0)


class PatchDataset(Dataset):
    """Match the notebook's crop order, augmentation, and center-crop validation."""

    def __init__(self, images: ImagePairs, names: list[str], patch: int, crops_per_image: int = 1,
                 train: bool = False):
        self.pairs = [images.load(name) for name in tqdm(names, desc="Load image pairs")]
        self.patch = patch
        self.crops_per_image = crops_per_image
        self.train = train

    def __len__(self) -> int:
        return len(self.pairs) * self.crops_per_image

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        dark, gt = self.pairs[index % len(self.pairs)]
        height, width = gt.shape[:2]
        size = self.patch
        if self.train:
            y = np.random.randint(0, height - size + 1)
            x = np.random.randint(0, width - size + 1)
        else:
            y, x = (height - size) // 2, (width - size) // 2
        dark, gt = dark[y:y + size, x:x + size], gt[y:y + size, x:x + size]
        if self.train:
            rotation = np.random.randint(4)
            dark, gt = np.rot90(dark, rotation), np.rot90(gt, rotation)
            if np.random.rand() < 0.5:
                dark, gt = dark[:, ::-1], gt[:, ::-1]
        return to_tensor(dark), to_tensor(gt)


def model_for(cfg: dict, device: torch.device) -> LowLightEnhancer:
    net = NAFUNet(
        width=cfg["width"], enc_blocks=cfg["enc_blocks"],
        middle_blocks=cfg["middle_blocks"], dec_blocks=cfg["dec_blocks"],
    )
    return LowLightEnhancer(net, gain=2.0 ** -cfg["ev_shift"], gamma=cfg["gamma"]).to(device)


def load_checkpoint(path: Path) -> dict:
    return torch.load(path, map_location="cpu", weights_only=True)


def save_checkpoint(state: dict, path: Path) -> None:
    temporary = path.with_suffix(".pt.tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def save_history(history: dict, path: Path) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(history, indent=2), encoding="utf-8")
    temporary.replace(path)


@torch.no_grad()
def center_crop_psnr(model: LowLightEnhancer, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    scores = []
    for dark, gt in loader:
        pred = model(dark.to(device)).clamp(0, 1)
        mse = (pred - gt.to(device)).square().mean().item()
        scores.append(10 * math.log10(1.0 / max(mse, 1e-10)))
    return float(np.mean(scores))


def train_variant(cfg: dict, split: dict, run_dir: Path, device: torch.device,
                  train_loader: DataLoader, val_loader: DataLoader) -> Path:
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=True)
    best_path, last_path = checkpoints / "best.pt", checkpoints / "last.pt"
    history_path = run_dir / "history.json"

    if cfg["loss"] == "original":
        criterion = RestorationLoss(ssim_weight=cfg["ssim_weight"]).to(device)
    else:
        criterion = ExposurePerceptualLoss(
            ssim_weight=cfg["ssim_weight"], exposure_weight=cfg["exposure_weight"],
            perceptual_weight=cfg["perceptual_weight"],
        ).to(device)

    # Seed after initializing the criterion, as in the reference notebook.
    random.seed(cfg["seed"])
    torch.manual_seed(cfg["seed"])
    model = model_for(cfg, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"], betas=(0.9, 0.9),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] * len(train_loader), eta_min=cfg["min_lr"],
    )

    start_epoch, best_psnr = 0, -math.inf
    history = {"epoch": [], "train_loss": [], "val_psnr": [], "lr": []}
    if last_path.is_file():
        state = load_checkpoint(last_path)
        if state["cfg"] != cfg or state["split"] != split:
            raise ValueError(f"Checkpoint settings differ in {run_dir}. Use a new --output-dir.")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch = state["epoch"] + 1
        best_psnr, history = state["best_psnr"], state["history"]
        del state
        save_history(history, history_path)
        print(f"{run_dir.name}: resuming after {start_epoch} completed epochs", flush=True)

    for epoch in range(start_epoch, cfg["epochs"]):
        np.random.seed(cfg["seed"] + epoch)
        train_loader.generator.manual_seed(cfg["seed"] + epoch)
        model.train()
        started = time.perf_counter()
        losses = []

        for dark, gt in tqdm(train_loader, desc=f"{run_dir.name} {epoch + 1}/{cfg['epochs']}", leave=False):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(dark.to(device)), gt.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            optimizer.step()
            scheduler.step()
            losses.append(loss.item())

        val_psnr = center_crop_psnr(model, val_loader, device)
        train_loss = float(np.mean(losses))
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["val_psnr"].append(val_psnr)
        history["lr"].append(scheduler.get_last_lr()[0])
        snapshot = {"model": model.state_dict(), "epoch": epoch, "val_psnr": val_psnr, "cfg": cfg}
        if val_psnr > best_psnr:
            best_psnr = val_psnr
            save_checkpoint(snapshot, best_path)
        save_checkpoint(
            {**snapshot, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
             "best_psnr": best_psnr, "history": history, "split": split}, last_path,
        )
        save_history(history, history_path)
        print(
            f"{run_dir.name} epoch {epoch + 1}: loss {train_loss:.4f}, "
            f"validation PSNR {val_psnr:.2f} dB, {time.perf_counter() - started:.0f}s",
            flush=True,
        )
    return best_path


@torch.no_grad()
def evaluate(checkpoint: Path, label: str, cfg: dict, images: ImagePairs, names: list[str],
             perceptual_metric: torch.nn.Module, device: torch.device) -> list[dict]:
    state = load_checkpoint(checkpoint)
    if state["cfg"] != cfg:
        raise ValueError(f"Unexpected settings in {checkpoint}. Check that it belongs to this run.")
    epoch = state["epoch"] + 1
    model = model_for(cfg, device)
    model.load_state_dict(state["model"])
    model.eval()
    del state
    rows = []
    for name in tqdm(names, desc=f"Full-image validation: {label}"):
        dark, gt = images.load(name)
        dark_tensor = to_tensor(dark)[None].to(device)
        gt_tensor = to_tensor(gt)[None].to(device)
        pred = tiled_inference(model, dark_tensor, cfg["tile"], cfg["tile_overlap"]).clamp(0, 1)
        pred_rgb = pred[0].permute(1, 2, 0).cpu().numpy()
        gt_rgb = gt.astype(np.float32) / 255.0
        rows.append({
            "model": label, "epoch": epoch, "image": name,
            "PSNR": peak_signal_noise_ratio(gt_rgb, pred_rgb, data_range=1.0),
            "SSIM": structural_similarity(gt_rgb, pred_rgb, channel_axis=-1, data_range=1.0),
            "LPIPS": perceptual_metric(2 * pred - 1, 2 * gt_tensor - 1).item(),
        })
    return rows


def write_reports(output_dir: Path, reference_history: dict) -> None:
    labels = {
        "reference": "Composite (reference)",
        "original": "Original loss",
        "composite_lr_3e-4": "Composite (LR 3e-4)",
    }
    histories = {labels["reference"]: reference_history}
    for folder in ("original", "composite_lr_3e-4"):
        path = output_dir / folder / "history.json"
        if path.is_file():
            histories[labels[folder]] = json.loads(path.read_text(encoding="utf-8"))

    fig, ax = plt.subplots(figsize=(9, 4))
    for label, history in histories.items():
        ax.plot(history["epoch"], history["val_psnr"], label=label)
    ax.set(xlabel="Epoch", ylabel="Validation PSNR [dB]", title="Center-crop validation PSNR")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "validation_psnr.png", dpi=150)
    plt.close(fig)

    metric_paths = [output_dir / folder / "validation_per_image.csv" for folder in labels]
    results = pd.concat([pd.read_csv(path) for path in metric_paths if path.is_file()], ignore_index=True)
    summary = results.groupby(["model", "epoch"], sort=False)[METRICS].mean().reset_index()
    summary.to_csv(output_dir / "validation_summary.csv", index=False)
    print("\nFull-image validation means")
    print(summary.round(4).to_string(index=False))
    print(f"\nResults: {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--step", choices=("all", "validate", "original", "lower-lr"), default="all")
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--composite-run", type=Path, help="Path to the completed composite run")
    parser.add_argument("--output-dir", type=Path, help="A separate directory for experiment outputs")
    args = parser.parse_args()

    project_dir = Path(__file__).resolve().parent
    runs_dir = Path(os.environ.get("SIGK_RUNS_DIR", project_dir / "runs"))
    source_dir = (args.composite_run or runs_dir / "composite").resolve()
    output_dir = (args.output_dir or source_dir.parent / "experiments").resolve()
    if output_dir == source_dir or output_dir.is_relative_to(source_dir):
        parser.error("Choose an output directory outside the composite run.")
    has_cuda = torch.cuda.is_available()
    if args.device == "cuda" and not has_cuda:
        parser.error("CUDA is unavailable. Connect to a Colab GPU runtime first.")
    device_name = args.device
    if device_name == "auto":
        device_name = "cuda" if has_cuda else "cpu"
    device = torch.device(device_name)
    print(f"Device: {device}", flush=True)

    reference_best = source_dir / "checkpoints" / "best.pt"
    reference_last = source_dir / "checkpoints" / "last.pt"
    for path in (reference_last, reference_best):
        if not path.is_file():
            parser.error(f"Missing {path}. Point --composite-run at the run with both checkpoints.")
    state = load_checkpoint(reference_last)
    cfg, split, reference_history = state["cfg"], state["split"], state["history"]
    if cfg.get("loss") != "composite":
        parser.error("The source checkpoint must be from the composite-loss run.")
    print(f"Reference: {state['epoch'] + 1}/{cfg['epochs']} epochs completed in {source_dir}", flush=True)
    del state

    validate_reference = args.step in ("all", "validate")
    variants = []
    if args.step in ("all", "original"):
        variants.append(("original", "original_best", {**cfg, "loss": "original"}))
    if args.step in ("all", "lower-lr"):
        if LOWER_LR >= cfg["lr"]:
            parser.error(f"Reference learning rate {cfg['lr']:g} is already <= {LOWER_LR:g}.")
        variants.append(("composite_lr_3e-4", "composite_lr_3e-4_best", {**cfg, "lr": LOWER_LR}))

    images = ImagePairs(project_dir / "data", cfg)
    images.check(split["val"])
    if variants:
        images.check(split["train"])
    perceptual_metric = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    if validate_reference:
        reference_out = output_dir / "reference"
        reference_out.mkdir(exist_ok=True)
        rows = evaluate(reference_best, "composite_best", cfg, images, split["val"], perceptual_metric, device)
        rows += evaluate(reference_last, "composite_last", cfg, images, split["val"], perceptual_metric, device)
        pd.DataFrame(rows).to_csv(reference_out / "validation_per_image.csv", index=False)
        write_reports(output_dir, reference_history)

    if variants:
        train_ds = PatchDataset(images, split["train"], cfg["patch"], cfg["crops_per_image"], train=True)
        val_ds = PatchDataset(images, split["val"], cfg["val_crop"])
        train_loader = DataLoader(
            train_ds, batch_size=cfg["batch_size"], shuffle=True, generator=torch.Generator(),
            num_workers=0, drop_last=True,
        )
        val_loader = DataLoader(val_ds, batch_size=1, shuffle=False)
        for folder, label, variant_cfg in variants:
            run_dir = output_dir / folder
            best_path = train_variant(variant_cfg, split, run_dir, device, train_loader, val_loader)
            rows = evaluate(best_path, label, variant_cfg, images, split["val"], perceptual_metric, device)
            pd.DataFrame(rows).to_csv(run_dir / "validation_per_image.csv", index=False)
            write_reports(output_dir, reference_history)


if __name__ == "__main__":
    main()
