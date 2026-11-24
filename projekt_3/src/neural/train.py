"""Training of the conditional GAN (LSGAN + mask-weighted L1 + mask BCE).

Run from the src/ directory:
    python -m neural.train --run_name relative_gan
    python -m neural.train --run_name no_gan --no_gan
    python -m neural.train --run_name raw --encoding raw
    python -m neural.train --run_name pixrel --pixel_rel_maps
    python -m neural.train --smoke            # smoke test on synthetic data

The test split is NEVER used here (only the train and val splits).
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from utils import config as cfg
from neural.dataset import PhongDataset, SyntheticDataset
from neural.encoding import cond_dim as encoding_dim
from neural.losses import d_loss_lsgan, generator_loss
from neural.models import CondPatchDiscriminator, ParamToImageGenerator, compose, count_params

LOSS_FIELDS = ["epoch", "d_loss", "g_total", "g_adv", "g_l1", "g_mask", "val_l1", "val_ssim", "seconds"]


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--runs_dir", type=Path, default=cfg.RUNS_DIR)
    ap.add_argument("--run_name", type=str, default="relative_gan")
    ap.add_argument("--encoding", choices=["raw", "relative"], default="relative")
    ap.add_argument("--pixel_rel_maps", action="store_true", help="per-pixel relative maps (x-u)/s in the generator")
    ap.add_argument("--no_gan", action="store_true", help="L1-only baseline (lambda_adv=0, no discriminator)")
    ap.add_argument("--alpha", type=float, default=20.0, help="mask weight in L1: w = 1 + alpha * mask")
    ap.add_argument("--lambda_adv", type=float, default=1.0)
    ap.add_argument("--lambda_l1", type=float, default=100.0)
    ap.add_argument("--lambda_mask", type=float, default=2.0)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--ema", type=float, default=0.999, help="EMA decay of the generator weights")
    ap.add_argument("--base_ch", type=int, default=512)
    ap.add_argument("--amp", action="store_true", help="mixed precision (CUDA only)")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--device", default="auto", help="auto | cpu | cuda | mps")
    ap.add_argument("--val_every", type=int, default=5)
    ap.add_argument("--sample_every", type=int, default=10)
    ap.add_argument("--num_workers", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="resume from runs/<name>/last.pt")
    ap.add_argument("--max_iters", type=int, default=None, help="(debug) iteration limit per epoch")
    ap.add_argument("--smoke", action="store_true", help="2 iterations on 16 synthetic examples")
    return ap


def pick_device(name: str) -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@torch.no_grad()
def ema_update(ema_model, model, decay: float):
    for pe, p in zip(ema_model.parameters(), model.parameters()):
        pe.mul_(decay).add_(p.detach(), alpha=1 - decay)


def to_uint8(img: torch.Tensor) -> np.ndarray:
    """(B,3,H,W) float in [0,1] -> (B,H,W,3) uint8 (quantisation as in a PNG)."""
    return (img.clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


@torch.no_grad()
def generate(G, cond: torch.Tensor) -> torch.Tensor:
    """Final image in [0,1] with a hard mask (as used at inference time)."""
    rgb, logit = G(cond)
    return compose(rgb, logit, hard=True)


@torch.no_grad()
def validate(G, loader, device) -> dict:
    """Masked L1 and SSIM on the validation split (images quantised to uint8 first)."""
    from skimage.metrics import structural_similarity
    G.eval()
    num, den, ssims = 0.0, 0.0, []
    for cond, real, mask in loader:
        cond, real, mask = cond.to(device), real.to(device), mask.to(device)
        fake_u8 = to_uint8(generate(G, cond))
        fake = torch.from_numpy(fake_u8).permute(0, 3, 1, 2).float().div(255.0).to(device)
        num += float(((fake - real).abs() * mask).sum())
        den += float(mask.sum() * 3)
        for a, b in zip(to_uint8(real), fake_u8):
            ssims.append(structural_similarity(a, b, channel_axis=2, data_range=255))
    G.train()
    return {"val_l1": num / max(den, 1.0), "val_ssim": float(np.mean(ssims))}


def save_samples(G, dataset, device, path: Path, n: int = 8):
    """Grid of validation samples: top row = reference, bottom row = network."""
    G.eval()
    n = min(n, len(dataset))
    cond = torch.stack([dataset[i][0] for i in range(n)]).to(device)
    real = torch.stack([dataset[i][1] for i in range(n)])
    with torch.no_grad():
        fake = generate(G, cond).cpu()
    G.train()
    grid = torch.nn.functional.interpolate(torch.cat([real, fake], dim=0), scale_factor=2, mode="nearest")
    save_image(grid, path, nrow=n)


def train(args) -> Path:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = pick_device(args.device)
    if args.smoke:
        args.epochs, args.batch_size, args.max_iters = 1, 8, 2
        args.val_every = args.sample_every = 1
        args.base_ch = min(args.base_ch, 128)
        if args.run_name == "relative_gan":
            args.run_name = "smoke"
    if args.no_gan:
        args.lambda_adv = 0.0
    if args.pixel_rel_maps and args.encoding != "relative":
        raise SystemExit("--pixel_rel_maps requires --encoding relative (needs u, v, s)")

    run_dir = Path(args.runs_dir) / args.run_name
    (run_dir / "samples").mkdir(parents=True, exist_ok=True)
    cond_size = encoding_dim(args.encoding)

    if args.smoke:
        train_ds = SyntheticDataset(16, args.encoding, seed=args.seed)
        val_ds = SyntheticDataset(8, args.encoding, seed=args.seed + 1)
    else:
        train_ds = PhongDataset(args.data_dir, "train", args.encoding)
        val_ds = PhongDataset(args.data_dir, "val", args.encoding)
    shuffle_gen = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              drop_last=len(train_ds) > args.batch_size,
                              num_workers=args.num_workers, generator=shuffle_gen)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    G = ParamToImageGenerator(cond_size, args.base_ch, args.pixel_rel_maps).to(device)
    G_ema = copy.deepcopy(G).eval()
    for p in G_ema.parameters():
        p.requires_grad_(False)
    opt_g = torch.optim.Adam(G.parameters(), lr=args.lr, betas=(0.5, 0.999))
    D, opt_d = None, None
    if not args.no_gan:
        D = CondPatchDiscriminator(cond_size).to(device)
        opt_d = torch.optim.Adam(D.parameters(), lr=args.lr, betas=(0.5, 0.999))
    print(f"device: {device} | G: {count_params(G):,} params | "
          f"D: {count_params(D) if D else 0:,} params | train={len(train_ds)} val={len(val_ds)}")

    use_amp = args.amp and device.type == "cuda"
    if args.amp and not use_amp:
        print("WARNING: --amp works only on CUDA - disabled")
    scaler_g = torch.amp.GradScaler("cuda", enabled=use_amp)
    scaler_d = torch.amp.GradScaler("cuda", enabled=use_amp)

    start_epoch, best_val = 0, float("inf")
    last_path, best_path = run_dir / "last.pt", run_dir / "best.pt"
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location=device)
        G.load_state_dict(ck["G"])
        G_ema.load_state_dict(ck["G_ema"])
        opt_g.load_state_dict(ck["opt_g"])
        if D is not None and opt_d is not None and ck.get("D") is not None:
            D.load_state_dict(ck["D"])
            opt_d.load_state_dict(ck["opt_d"])
        start_epoch, best_val = ck["epoch"] + 1, ck["best_val"]
        print(f"resumed from epoch {start_epoch}")

    config = {**{k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
              "cond_dim": cond_size, "device": str(device)}
    (run_dir / "config.json").write_text(json.dumps(config, indent=2))
    log_path = run_dir / "losses.csv"
    if not (args.resume and log_path.exists()):
        with open(log_path, "w", newline="") as f:
            csv.DictWriter(f, LOSS_FIELDS).writeheader()

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        sums = {"d_loss": 0.0, "g_total": 0.0, "g_adv": 0.0, "g_l1": 0.0, "g_mask": 0.0}
        n_it = 0
        for it, (cond, real, mask) in enumerate(train_loader):
            if args.max_iters is not None and it >= args.max_iters:
                break
            cond, real, mask = cond.to(device), real.to(device), mask.to(device)

            with torch.autocast(device.type, enabled=use_amp):
                rgb, logit = G(cond)
                fake = compose(rgb, logit)                      # soft mask during training

            # --- discriminator step ---
            d_loss = torch.zeros((), device=device)
            if D is not None and opt_d is not None:
                with torch.autocast(device.type, enabled=use_amp):
                    d_loss = d_loss_lsgan(D(real, cond), D(fake.detach(), cond))
                opt_d.zero_grad(set_to_none=True)
                scaler_d.scale(d_loss).backward()
                scaler_d.step(opt_d)
                scaler_d.update()

            # --- generator step ---
            with torch.autocast(device.type, enabled=use_amp):
                d_fake = D(fake, cond) if D is not None else None
                g_total, parts = generator_loss(fake.float(), logit.float(), real, mask,
                                                d_fake.float() if d_fake is not None else None,
                                                args.lambda_adv, args.lambda_l1, args.lambda_mask, args.alpha)
            opt_g.zero_grad(set_to_none=True)
            scaler_g.scale(g_total).backward()
            scaler_g.step(opt_g)
            scaler_g.update()
            ema_update(G_ema, G, args.ema)

            sums["d_loss"] += d_loss.item()
            sums["g_total"] += g_total.item()
            for k in ("g_adv", "g_l1", "g_mask"):
                sums[k] += parts[k]
            n_it += 1

        row = {"epoch": epoch, **{k: v / max(n_it, 1) for k, v in sums.items()},
               "val_l1": "", "val_ssim": "", "seconds": round(time.time() - t0, 1)}
        is_last = epoch == args.epochs - 1
        if (epoch + 1) % args.val_every == 0 or is_last:
            m = validate(G_ema, val_loader, device)
            row.update(val_l1=m["val_l1"], val_ssim=m["val_ssim"])
            if m["val_l1"] < best_val:                      # checkpoint selection: masked val L1
                best_val = m["val_l1"]
                torch.save({"G_ema": G_ema.state_dict(), "config": config, "epoch": epoch, **m}, best_path)
            print(f"epoch {epoch + 1}/{args.epochs}  g_l1={row['g_l1']:.4f}  val_l1={m['val_l1']:.4f}  "
                  f"val_ssim={m['val_ssim']:.4f}")
        if (epoch + 1) % args.sample_every == 0 or is_last:
            save_samples(G_ema, val_ds, device, run_dir / "samples" / f"epoch_{epoch + 1:04}.png")
        with open(log_path, "a", newline="") as f:
            csv.DictWriter(f, LOSS_FIELDS).writerow(row)
        torch.save({"G": G.state_dict(), "G_ema": G_ema.state_dict(), "opt_g": opt_g.state_dict(),
                    "D": D.state_dict() if D else None, "opt_d": opt_d.state_dict() if opt_d else None,
                    "epoch": epoch, "best_val": best_val, "config": config}, last_path)
    print(f"done: {run_dir}")
    return run_dir


def main(argv=None):
    train(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
