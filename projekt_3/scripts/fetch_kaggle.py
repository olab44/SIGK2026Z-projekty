"""Wait for a Kaggle kernel to finish and download its small outputs into projekt_3/.

    python scripts/fetch_kaggle.py <kaggle-user>/sigk3-gan-v1

The kernel writes to /kaggle/working/runs/<run_name>/, so the files land in projekt_3/runs/<run_name>/
(best.pt, losses.csv, config.json, samples/). last.pt (optimizer state, large) is skipped.
Requires Kaggle credentials (see README); uses `uvx --with six kaggle`.
"""
import _bootstrap  # noqa: F401  (sets sys.path)

import argparse
import re
import subprocess
import sys
import time

from utils import config as cfg

KAGGLE = ["uvx", "--with", "six", "kaggle"]
FINAL = ("COMPLETE", "ERROR", "CANCEL")


def kernel_status(kernel: str) -> str:
    out = subprocess.run([*KAGGLE, "kernels", "status", kernel], capture_output=True, text=True)
    text = out.stdout + out.stderr
    m = re.search(r"\b(QUEUED|RUNNING|COMPLETE|ERROR|CANCEL\w*)\b", text.upper())
    print(text.strip())
    return m.group(1) if m else "UNKNOWN"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("kernel", help="<kaggle-user>/<kernel-slug>")
    ap.add_argument("--poll", type=int, default=120, help="seconds between status checks")
    ap.add_argument("--pattern", default=r"(best\.pt|config\.json|losses\.csv|png)$",
                    help="regex for the files to download (matched on the whole path)")
    ap.add_argument("--out", default=str(cfg.PROJECT_ROOT))
    args = ap.parse_args(argv)

    while True:
        status = kernel_status(args.kernel)
        if status.startswith(FINAL):
            break
        time.sleep(args.poll)
    if not status.startswith("COMPLETE"):
        sys.exit(f"kernel finished with status {status}; Kaggle deletes the output of failed kernels")
    subprocess.run([*KAGGLE, "kernels", "output", args.kernel, "-p", args.out, "--file-pattern", args.pattern],
                   check=True)
    print(f"downloaded to {args.out}/runs/")


if __name__ == "__main__":
    main()
