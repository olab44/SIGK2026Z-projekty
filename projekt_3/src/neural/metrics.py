"""Quality metrics: FLIP, LPIPS, SSIM and the Hausdorff distance between Canny edge images.

All metrics take the reference and the generated image as RGB uint8 arrays of shape (H, W, 3).
The network output is quantised to uint8 (as in a PNG) BEFORE the metrics are computed.
"""
from __future__ import annotations

import cv2
import flip_evaluator as flip
import lpips
import numpy as np
import torch
from scipy.spatial.distance import directed_hausdorff
from skimage.metrics import structural_similarity

# Canny thresholds. The unlit side of the sphere has brightness ~8/255 (0.1 * 0.3 ambient), so with
# typical thresholds (100, 200) its contour disappears; hence the lower defaults (20, 60).
DEFAULT_CANNY = (20, 60)
METRIC_NAMES = ["flip", "lpips", "ssim", "hausdorff"]


def _check(img: np.ndarray):
    assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3, "expected RGB uint8 (H, W, 3)"


def ssim(ref: np.ndarray, gen: np.ndarray) -> float:
    _check(ref)
    _check(gen)
    return float(structural_similarity(ref, gen, channel_axis=2, data_range=255))


def flip_error(ref: np.ndarray, gen: np.ndarray):
    """LDR-FLIP via flip_evaluator.evaluate(reference, test, "LDR").

    API checked in flip_evaluator/flip_python_api.py: HxWx3 float32 arrays in [0, 1] (sRGB); returns
    (error map, mean error, parameters); with applyMagma=False the map has shape HxWx1.
    Returns (mean error, error map HxW in [0, 1]).
    """
    _check(ref)
    _check(gen)
    r = np.ascontiguousarray(ref, dtype=np.float32) / 255.0
    g = np.ascontiguousarray(gen, dtype=np.float32) / 255.0
    err_map, mean_err, _ = flip.evaluate(r, g, "LDR", applyMagma=False)
    return float(mean_err), np.asarray(err_map)[..., 0]


def edge_points(img: np.ndarray, t_low: float, t_high: float) -> np.ndarray:
    """(row, col) coordinates of Canny edge pixels of the grayscale image."""
    edges = cv2.Canny(cv2.cvtColor(img, cv2.COLOR_RGB2GRAY), t_low, t_high)
    return np.argwhere(edges > 0).astype(np.float64)


def hausdorff_edges(ref: np.ndarray, gen: np.ndarray, t_low: float = DEFAULT_CANNY[0],
                    t_high: float = DEFAULT_CANNY[1]):
    """Symmetric Hausdorff distance [px] between the Canny edge images.

    Edge cases: both edge sets empty -> 0; exactly one empty -> penalty = image diagonal (~181 px).
    Returns (distance, whether the penalty was applied).
    """
    _check(ref)
    _check(gen)
    a, b = edge_points(ref, t_low, t_high), edge_points(gen, t_low, t_high)
    if len(a) == 0 and len(b) == 0:
        return 0.0, False
    if len(a) == 0 or len(b) == 0:
        return float(np.hypot(*ref.shape[:2])), True
    return float(max(directed_hausdorff(a, b)[0], directed_hausdorff(b, a)[0])), False


class MetricSuite:
    """Holds the LPIPS network (alex) and computes all four metrics for an image pair."""

    def __init__(self, device: str = "cpu", canny=DEFAULT_CANNY):
        self.device = torch.device(device)
        self.canny = tuple(canny)
        self._lpips = lpips.LPIPS(net="alex", verbose=False).to(self.device).eval()

    @torch.no_grad()
    def lpips(self, ref: np.ndarray, gen: np.ndarray) -> float:
        """LPIPS (alex): NCHW input in [-1, 1]."""
        _check(ref)
        _check(gen)

        def to_tensor(x):
            return (torch.from_numpy(x).permute(2, 0, 1)[None].float() / 127.5 - 1.0).to(self.device)

        return float(self._lpips(to_tensor(ref), to_tensor(gen)).item())

    def __call__(self, ref: np.ndarray, gen: np.ndarray) -> dict:
        haus, penalty = hausdorff_edges(ref, gen, *self.canny)
        return {"flip": flip_error(ref, gen)[0], "lpips": self.lpips(ref, gen), "ssim": ssim(ref, gen),
                "hausdorff": haus, "hausdorff_penalty": int(penalty)}


def summarize(values: dict) -> dict:
    """{metric: array} -> {metric: {mean, std, median}}."""
    return {k: {"mean": float(np.mean(v)), "std": float(np.std(v)), "median": float(np.median(v))}
            for k, v in values.items()}
