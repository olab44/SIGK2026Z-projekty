"""Kaggle job: train the conditional GAN on the dataset committed in git (no uploads needed)."""
import subprocess
import sys

import torch

REPO = "https://github.com/olab44/SIGK2026Z-projekty.git"
BRANCH = "main"
CLONE = "/tmp/repo"
# One entry per training run; extra CLI flags go after the name (e.g. "--no_gan").
RUNS = [("relative_gan", [])]

assert torch.cuda.is_available(), "no GPU: check the phone verification of the Kaggle account"

subprocess.run(["git", "clone", "--depth", "1", "--branch", BRANCH, REPO, CLONE], check=True)
print("commit", subprocess.run(["git", "-C", CLONE, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip())

# torch, pandas, scikit-image and matplotlib are preinstalled on Kaggle; do not pip install requirements.txt.
for name, flags in RUNS:
    subprocess.run([sys.executable, "-m", "neural.train", "--run_name", name, "--runs_dir", "/kaggle/working/runs",
                    "--data_dir", f"{CLONE}/projekt_3/data/phong", "--amp", "--num_workers", "2", *flags],
                   cwd=f"{CLONE}/projekt_3/src", check=True)
