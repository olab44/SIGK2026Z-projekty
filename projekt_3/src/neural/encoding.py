"""Encoding of raw world parameters (a params.csv row) into the conditioning vector of the network.

The CSV stores RAW parameters; the encoding is done here (hint 1 of the assignment).

Modes:
  raw      (D=10): obj/20, light/20, kd/255, (n-3)/17
  relative (D=11, default): quantities relative to the camera and the sphere:
      u, v     - sphere centre in NDC,                u = x/(z t), v = y/(z t)
      s        - apparent sphere radius in NDC,       s = R/(z t)
      l (3)    - unit direction from the sphere centre to the light, in the camera frame
      |l|/70   - light-sphere distance (max sqrt(3)*40 ~ 69.3)
      kd/255   (3), (n-3)/17
  where (x, y, z) is the sphere centre in the camera frame (z = depth) and t = tan(fov/2).
"""
from __future__ import annotations

import numpy as np

from utils import config as cfg
import scene_sampling as ss

COND_DIM = {"raw": 10, "relative": 11}
MAX_LIGHT_DISTANCE = 70.0  # > sqrt(3) * 40


def row_from_scene(scene: dict) -> dict:
    """Scene dict from sample_scene -> flat row with the params.csv column names."""
    o, l, d = scene["obj"], scene["light"], scene["diffuse"]
    return {"obj_x": o[0], "obj_y": o[1], "obj_z": o[2], "light_x": l[0], "light_y": l[1], "light_z": l[2],
            "kd_r": d[0], "kd_g": d[1], "kd_b": d[2], "shininess": scene["shininess"]}


def _unpack(row):
    obj = np.array([row["obj_x"], row["obj_y"], row["obj_z"]], dtype=np.float64)
    light = np.array([row["light_x"], row["light_y"], row["light_z"]], dtype=np.float64)
    kd = np.array([row["kd_r"], row["kd_g"], row["kd_b"]], dtype=np.float64)
    return obj, light, kd, float(row["shininess"])


def encode(params_row, mode: str = "relative") -> np.ndarray:
    """Row (dict / pandas.Series) -> np.float32[D]."""
    obj, light, kd, n = _unpack(params_row)
    lo, hi = cfg.SHININESS_RANGE
    n_enc = (n - lo) / (hi - lo)
    kd_enc = kd / 255.0
    if mode == "raw":
        vec = np.concatenate([obj / 20.0, light / 20.0, kd_enc, [n_enc]])
    elif mode == "relative":
        u, v = ss.project_center_ndc(obj)
        s = ss.apparent_radius_ndc(obj)
        to_light = ss.direction_to_camera_frame(light - obj)
        dist = np.linalg.norm(to_light)
        direction = to_light / max(dist, 1e-8)
        vec = np.concatenate([[u, v, s], direction, [dist / MAX_LIGHT_DISTANCE], kd_enc, [n_enc]])
    else:
        raise ValueError(f"unknown encoding mode: {mode!r}")
    return vec.astype(np.float32)


def cond_dim(mode: str) -> int:
    return COND_DIM[mode]
