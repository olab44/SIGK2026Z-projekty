"""Random train/val/test split (seeded) -> split.json.

The test split must never be used for training or checkpoint selection.
"""
import _bootstrap  # noqa: F401

import argparse
import json
from pathlib import Path

import numpy as np

from utils import config as cfg


def make_split(n: int, test_size: int, val_size: int, seed: int) -> dict:
    perm = np.random.default_rng(seed).permutation(n)
    test = perm[:test_size]
    val = perm[test_size:test_size + val_size]
    train = perm[test_size + val_size:]
    return {"seed": seed, "train": sorted(map(int, train)), "val": sorted(map(int, val)),
            "test": sorted(map(int, test))}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data_dir", type=Path, default=cfg.DATA_DIR)
    ap.add_argument("--num_images", type=int, default=cfg.NUM_IMAGES)
    ap.add_argument("--test_size", type=int, default=cfg.TEST_SIZE)
    ap.add_argument("--val_size", type=int, default=cfg.VAL_SIZE)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    args = ap.parse_args(argv)

    split = make_split(args.num_images, args.test_size, args.val_size, args.seed)
    out = args.data_dir / "split.json"
    out.write_text(json.dumps(split))
    print(f"{out}: train={len(split['train'])} val={len(split['val'])} test={len(split['test'])}")


if __name__ == "__main__":
    main()
