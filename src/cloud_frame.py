"""Point-cloud frame utilities for coherent rendering and view alignment.

This module fixes a key mismatch in the original pipeline:
Point-E point clouds live in their own object-centric coordinates, but the
semantic painting code was projecting them using the *mesh simulator* camera
poses (world coordinates). That produced incoherent "starfield" renders.

Here we instead:
- Normalize clouds into a stable object frame (centered + optionally scaled).
- Define an orbital camera *in that same frame* using the known view index.
- Estimate a global yaw that makes the cloud's silhouette match the conditioning
  image silhouette, ensuring "front-face alignment" w.r.t. the input view.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple

import cv2
import numpy as np

from .segmentation import render_point_cloud_view
from .simulator import generate_orbital_poses, pose_to_matrix, Pose


@dataclass(frozen=True)
class CloudNormalization:
    """Affine normalization applied to Point-E clouds."""

    center: np.ndarray  # (3,) translation subtracted from raw points
    scale: float  # scalar applied after centering (raw_centered * scale)


def _rotation_z(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _angle_wrap_deg(deg: float) -> float:
    """Wrap to [-180, 180)."""
    x = float(deg)
    x = (x + 180.0) % 360.0 - 180.0
    return x


def compute_object_mask(rgb: np.ndarray, *, threshold: int = 5) -> np.ndarray:
    """Binary object mask from an RGB render.

    Robust to both dark and light backgrounds.

    Why:
    - The yaw-alignment loop relies on a stable silhouette mask.
    - A simple `gray > threshold` assumption only works for black backgrounds.
    - Many users set white backgrounds for nicer figures, which can cause
      "starfield" misalignment failures.

    Strategy:
    - Use Otsu threshold on grayscale to separate object vs. background.
    - If the resulting foreground occupies most of the image, flip it.
    - Fall back to a simple threshold if Otsu degenerates.
    """
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"Expected RGB image (H,W,3), got {rgb.shape}")

    gray = rgb.mean(axis=2).astype(np.uint8)
    try:
        # Otsu works well for typical solid-background renders.
        _, m = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        mask = m.astype(np.uint8)
        # If mask covers almost everything, invert (object should be smaller than bg).
        if float(mask.mean()) > 0.6:
            mask = (1 - mask).astype(np.uint8)
        # Degenerate case: fallback.
        if mask.sum() < 10:
            mask = (gray > int(threshold)).astype(np.uint8)
        return mask
    except Exception:
        return (gray > int(threshold)).astype(np.uint8)


def normalize_clouds_to_bounds(
    clouds: List[np.ndarray],
    *,
    grid_bounds: float,
    margin: float = 0.95,
    center_method: str = "median",
) -> Tuple[List[np.ndarray], CloudNormalization]:
    """Center and (optionally) scale clouds so they fit inside [-grid_bounds, grid_bounds].

    We use a robust center so that protrusions/outliers (e.g., hallucinated handles)
    do not dominate the normalization.
    """
    if not clouds:
        return [], CloudNormalization(center=np.zeros(3, dtype=np.float32), scale=1.0)

    ref = np.asarray(clouds[0], dtype=np.float32)
    if ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N,3), got {ref.shape}")

    if center_method not in {"median", "mean"}:
        raise ValueError("center_method must be 'median' or 'mean'")
    center = np.median(ref, axis=0) if center_method == "median" else ref.mean(axis=0)
    center = np.asarray(center, dtype=np.float32).reshape(3)

    centered = [np.asarray(pc, dtype=np.float32) - center[None, :] for pc in clouds]
    all_pts = np.concatenate([pc for pc in centered if pc.size > 0], axis=0) if centered else ref * 0
    if all_pts.size == 0:
        return centered, CloudNormalization(center=center, scale=1.0)

    # Robust extent estimate: ignore extreme outliers that would shrink everything.
    abs_xyz = np.abs(all_pts)
    robust_max = float(np.quantile(abs_xyz, 0.995))
    robust_max = max(robust_max, 1e-6)
    target = float(margin) * float(grid_bounds)
    scale = 1.0 if robust_max <= target else (target / robust_max)

    normalized = [pc * float(scale) for pc in centered]
    return normalized, CloudNormalization(center=center, scale=float(scale))


def estimate_orbit_distance_for_cloud(
    points: np.ndarray,
    intrinsics,
    *,
    target_fill: float = 0.65,
    margin: float = 1.15,
    clamp: Tuple[float, float] = (0.15, 5.0),
) -> float:
    """Pick a camera radius so the cloud occupies ~target_fill of the viewport."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N,3), got {pts.shape}")
    if pts.size == 0:
        return float(np.mean(clamp))

    # Robust extents: raw min/max can be dominated by a handful of outliers
    # (or sparse stray points) which makes the estimated camera radius too large
    # and causes the object to appear "far away" in renders.
    mins = np.quantile(pts, 0.01, axis=0)
    maxs = np.quantile(pts, 0.99, axis=0)
    ext = maxs - mins
    obj_radius = 0.5 * float(np.max(ext))
    obj_radius = max(obj_radius, 1e-4)

    min_dim = float(min(getattr(intrinsics, "width"), getattr(intrinsics, "height")))
    f = float(min(getattr(intrinsics, "fx"), getattr(intrinsics, "fy")))
    half_px = 0.5 * float(np.clip(target_fill, 0.1, 0.95)) * min_dim
    alpha = math.atan2(half_px, f)
    dist = obj_radius / max(math.tan(alpha), 1e-6)
    dist *= float(margin)
    lo, hi = float(clamp[0]), float(clamp[1])
    return float(np.clip(dist, lo, hi))


def make_cloud_orbital_poses(
    *,
    num_views: int,
    elevation_deg: float,
    radius: float,
) -> List[Pose]:
    # generate_orbital_poses already matches the simulator view indexing.
    return generate_orbital_poses(int(num_views), float(radius), float(elevation_deg), look_at=np.zeros(3, dtype=np.float32))


def estimate_yaw_align_to_mask(
    points: np.ndarray,
    *,
    input_mask: np.ndarray,
    base_cam_to_world: np.ndarray,
    intrinsics,
    prior_yaw_deg: Optional[float] = None,
    prior_window_deg: float = 30.0,
    coarse_step_deg: float = 10.0,
    fine_step_deg: float = 2.0,
    point_radius_px: int = 2,
    blur_sigma: float = 0.8,
    render_threshold: int = 10,
) -> float:
    """Estimate a global yaw (about +Z) aligning the cloud silhouette to the input mask."""
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"Expected point cloud shape (N,3), got {pts.shape}")

    in_mask = (np.asarray(input_mask) > 0).astype(np.uint8)
    if in_mask.ndim != 2:
        raise ValueError(f"Expected input_mask shape (H,W), got {in_mask.shape}")

    # Light morphology to reduce sensitivity to point sparsity.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    in_mask = cv2.morphologyEx(in_mask, cv2.MORPH_CLOSE, k, iterations=1)

    def _iou(a: np.ndarray, b: np.ndarray) -> float:
        a = (a > 0)
        b = (b > 0)
        inter = float(np.logical_and(a, b).sum())
        union = float(np.logical_or(a, b).sum())
        return inter / union if union > 1e-6 else 0.0

    def _render_mask(yaw_deg: float) -> np.ndarray:
        R = _rotation_z(float(yaw_deg))
        pts_r = (pts @ R.T).astype(np.float32)
        img = render_point_cloud_view(
            pts_r,
            base_cam_to_world,
            intrinsics,
            point_radius_px=int(point_radius_px),
            blur_sigma=float(blur_sigma),
        )
        gray = img[..., 0]
        mask = (gray > int(render_threshold)).astype(np.uint8)
        # Close small holes so IoU tracks silhouette overlap more than point density.
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k, iterations=1)

    if prior_yaw_deg is None:
        coarse_yaws = np.arange(-180.0, 180.0, float(coarse_step_deg), dtype=np.float32)
    else:
        prior = float(prior_yaw_deg)
        w = float(max(prior_window_deg, coarse_step_deg))
        coarse_yaws = np.arange(prior - w, prior + w + 1e-6, float(coarse_step_deg), dtype=np.float32)

    best_yaw = 0.0
    best_score = -1.0

    # Tie-breaking matters: for near-symmetric silhouettes (e.g., handle fully occluded),
    # many yaws yield identical IoU. The previous implementation kept the *first*
    # maximum (often -180deg), which can flip view-index mapping and degrade NBV.
    # We instead prefer the yaw closest to the prior (or 0deg when no prior).
    ref_yaw = float(prior_yaw_deg) if prior_yaw_deg is not None else 0.0
    eps = 1e-9
    for yaw in coarse_yaws:
        score = _iou(_render_mask(float(yaw)), in_mask)
        if (score > best_score + eps) or (
            abs(score - best_score) <= eps
            and abs(_angle_wrap_deg(float(yaw) - ref_yaw)) < abs(_angle_wrap_deg(float(best_yaw) - ref_yaw)) - 1e-9
        ):
            best_score = score
            best_yaw = float(yaw)

    # Fine search around the best coarse yaw.
    fine_step = float(max(fine_step_deg, 0.5))
    fine_yaws = np.arange(best_yaw - float(coarse_step_deg), best_yaw + float(coarse_step_deg) + 1e-6, fine_step, dtype=np.float32)
    for yaw in fine_yaws:
        score = _iou(_render_mask(float(yaw)), in_mask)
        if (score > best_score + eps) or (
            abs(score - best_score) <= eps
            and abs(_angle_wrap_deg(float(yaw) - ref_yaw)) < abs(_angle_wrap_deg(float(best_yaw) - ref_yaw)) - 1e-9
        ):
            best_score = score
            best_yaw = float(yaw)

    return _angle_wrap_deg(best_yaw)


def rotate_clouds_yaw(clouds: List[np.ndarray], yaw_deg: float) -> List[np.ndarray]:
    if not clouds:
        return []
    R = _rotation_z(float(yaw_deg))
    return [(np.asarray(pc, dtype=np.float32) @ R.T).astype(np.float32) for pc in clouds]
