"""Scene sampling (with rejection) and camera geometry for the neural Phong renderer project.

The module depends only on numpy (no OpenGL, no PyTorch), so it is shared by the data
generator (PhongWindow) and the training code.

Camera-frame convention used throughout the project:
    x - right, y - up, z - DEPTH (positive in front of the camera).
This is the OpenGL camera frame with the sign of z flipped, so the NDC coordinates are simply
u = x / (z * t), v = y / (z * t), where t = tan(fov / 2).
Image convention: u grows to the right, v grows upwards.
"""
from __future__ import annotations

import math
from collections import Counter

import numpy as np

from utils import config as cfg


# ---------------------------------------------------------------------------
# Camera geometry
# ---------------------------------------------------------------------------

def _normalize(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def camera_basis():
    """Return (right, up, forward) of the camera, computed exactly like pyrr's `create_look_at`."""
    eye = np.asarray(cfg.CAMERA_POSITION, dtype=np.float64)
    target = np.asarray(cfg.CAMERA_TARGET, dtype=np.float64)
    up = np.asarray(cfg.CAMERA_UP, dtype=np.float64)
    forward = _normalize(target - eye)
    right = _normalize(np.cross(forward, up))
    true_up = _normalize(np.cross(right, forward))
    return right, true_up, forward


def tan_half_fov() -> float:
    return math.tan(math.radians(cfg.FOV_DEG) / 2.0)


def world_to_camera(p) -> np.ndarray:
    """World point -> (x, y, z) in the camera frame (z = depth, > 0 in front of the camera)."""
    right, up, forward = camera_basis()
    d = np.asarray(p, dtype=np.float64) - np.asarray(cfg.CAMERA_POSITION, dtype=np.float64)
    return np.array([d @ right, d @ up, d @ forward])


def direction_to_camera_frame(d) -> np.ndarray:
    """World-space direction vector -> camera frame (rotation only, no translation)."""
    right, up, forward = camera_basis()
    d = np.asarray(d, dtype=np.float64)
    return np.array([d @ right, d @ up, d @ forward])


def sphere_fully_visible(p, margin: float = 0.0) -> bool:
    """Whether a sphere (radius R) centred at p lies entirely inside the view frustum.

    Condition: the distance from the centre to each of the 4 side planes is >= R + margin
    (aspect = 1, so both half-angles have tangent t = tan(fov/2)), and
    near + R < depth < far - R.
    """
    x, y, z = world_to_camera(p)
    t = tan_half_fov()
    need = cfg.SPHERE_RADIUS + margin
    norm = math.sqrt(1.0 + t * t)
    # Planes: z*t - x = 0, z*t + x = 0, z*t - y = 0, z*t + y = 0.
    dists = ((z * t - x) / norm, (z * t + x) / norm, (z * t - y) / norm, (z * t + y) / norm)
    if min(dists) < need:
        return False
    return (cfg.NEAR + cfg.SPHERE_RADIUS) < z < (cfg.FAR - cfg.SPHERE_RADIUS)


def project_center_ndc(p) -> tuple[float, float]:
    """Sphere centre in NDC: (u, v), u to the right, v up (within [-1, 1] for visible spheres)."""
    x, y, z = world_to_camera(p)
    t = tan_half_fov()
    return float(x / (z * t)), float(y / (z * t))


def apparent_radius_ndc(p) -> float:
    """Apparent sphere radius in NDC: s = R / (z * t) (approximation)."""
    z = world_to_camera(p)[2]
    return float(cfg.SPHERE_RADIUS / (z * tan_half_fov()))


# ---------------------------------------------------------------------------
# Rejection sampling
# ---------------------------------------------------------------------------

def check_scene(obj, light, margin=None, min_light_distance=None, max_camera_distance=...):
    """Return None if the scene is acceptable, otherwise the name of the rejection reason."""
    margin = cfg.VISIBILITY_MARGIN if margin is None else margin
    min_light_distance = cfg.MIN_LIGHT_DISTANCE if min_light_distance is None else min_light_distance
    if max_camera_distance is ...:
        max_camera_distance = cfg.MAX_CAMERA_DISTANCE

    z = world_to_camera(obj)[2]
    if not (z > cfg.NEAR + cfg.SPHERE_RADIUS):          # (b) sphere behind the camera / in front of near
        return "depth"
    if not sphere_fully_visible(obj, margin):             # (a) sphere fully inside the frustum
        return "frustum"
    if np.linalg.norm(np.asarray(light) - np.asarray(obj)) < min_light_distance:  # (c) light too close
        return "light_too_close"
    if max_camera_distance is not None:
        if np.linalg.norm(np.asarray(obj) - np.asarray(cfg.CAMERA_POSITION)) > max_camera_distance:
            return "camera_distance"
    return None


def sample_scene(rng: np.random.Generator, stats: Counter | None = None, max_attempts: int = 1_000_000) -> dict:
    """Sample a scene from the assignment ranges and accept it only if it satisfies the constraints.

    Returns dict(obj, light, diffuse (int, 0..255), shininess, attempts).
    `stats` (a Counter) is updated with the number of rejections per reason and the keys
    "accepted" and "attempts".
    """
    lo, hi = cfg.POS_RANGE
    attempts = 0
    while True:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError("sample_scene: max_attempts exceeded - constraints too strict?")
        # Round to 1e-6 so that the CSV describes exactly the rendered scene.
        obj = np.round(rng.uniform(lo, hi, size=3), 6)
        light = np.round(rng.uniform(lo, hi, size=3), 6)
        diffuse = rng.integers(cfg.DIFFUSE_RANGE[0], cfg.DIFFUSE_RANGE[1] + 1, size=3)
        shininess = float(np.round(rng.uniform(*cfg.SHININESS_RANGE), 6))

        reason = check_scene(obj, light)
        if stats is not None:
            stats["attempts"] += 1
            stats[reason or "accepted"] += 1
        if reason is None:
            return {
                "obj": obj,
                "light": light,
                "diffuse": diffuse.astype(int),
                "shininess": shininess,
                "attempts": attempts,
            }
