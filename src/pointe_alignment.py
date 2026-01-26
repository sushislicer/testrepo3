"""Robust multi-seed alignment utilities for Point-E point clouds.

Goal: Given K Point-E hypotheses conditioned on the same input view, enforce that
the *front/visible* body surfaces align across seeds so that variance reflects
true ambiguity (e.g., occluded handles) instead of global pose jitter.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class AlignmentParams:
    max_yaw_deg: float = 30.0
    yaw_step_deg: float = 5.0
    trim_ratio: float = 0.7
    core_ratio: float = 1.0
    surface_grid: int = 96
    max_points: int = 2048


def _rotation_z(deg: float) -> np.ndarray:
    rad = math.radians(float(deg))
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _robust_center(points: np.ndarray) -> np.ndarray:
    # Median is robust to outliers such as hallucinated handles.
    return np.median(points, axis=0).astype(np.float32)


def _robust_scale(points: np.ndarray, center: np.ndarray, percentile: float = 90.0) -> float:
    d = np.linalg.norm(points - center[None, :], axis=1)
    s = float(np.percentile(d, percentile))
    return s if s > 1e-6 else 1.0


def _keep_core_points(points: np.ndarray, ratio: float) -> np.ndarray:
    """Keep the inner 'body' points (drops protrusions like handles).

    We use radial distance from the robust center. For mugs, handle points tend
    to be farther from the center than the main cylindrical body.
    """
    if points.shape[0] == 0:
        return points
    ratio = float(np.clip(ratio, 0.05, 1.0))
    if ratio >= 0.999:
        return points
    center = _robust_center(points)
    d = np.linalg.norm(points - center[None, :], axis=1)
    # Combine a quantile cutoff with a robust MAD cutoff so we can drop extreme
    # protrusions even when they represent a non-trivial fraction of points.
    thresh_q = float(np.quantile(d, ratio))
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-6
    thresh_mad = med + 3.5 * mad
    thresh = min(thresh_q, thresh_mad)
    keep = d <= thresh
    # Ensure non-empty output.
    if not np.any(keep):
        keep[np.argmin(d)] = True
    return points[keep]


def _extract_front_surface(points: np.ndarray, grid_res: int) -> np.ndarray:
    """Approximate the visible 'front surface' under an orthographic +Y view.

    Assumes the conditioning/front direction in the Point-E coordinate frame is
    approximately +Y. For each (x,z) bin we keep the point with the largest y.
    This removes points on the back side (including occluded handles) without
    relying on explicit segmentation.
    """
    if points.shape[0] == 0:
        return points
    grid_res = int(np.clip(grid_res, 16, 512))

    x = points[:, 0].astype(np.float32)
    y = points[:, 1].astype(np.float32)
    z = points[:, 2].astype(np.float32)

    # Robust bounds in the projected plane.
    x0, x1 = np.quantile(x, [0.01, 0.99]).astype(np.float32)
    z0, z1 = np.quantile(z, [0.01, 0.99]).astype(np.float32)
    if float(x1 - x0) < 1e-6:
        x0, x1 = float(x.min()), float(x.max() + 1e-6)
    if float(z1 - z0) < 1e-6:
        z0, z1 = float(z.min()), float(z.max() + 1e-6)

    # Bin coordinates to [0, grid_res-1].
    u = np.floor((x - x0) / max(float(x1 - x0), 1e-6) * (grid_res - 1)).astype(np.int32)
    v = np.floor((z - z0) / max(float(z1 - z0), 1e-6) * (grid_res - 1)).astype(np.int32)
    u = np.clip(u, 0, grid_res - 1)
    v = np.clip(v, 0, grid_res - 1)
    flat = u + grid_res * v

    best_y = np.full(grid_res * grid_res, -np.inf, dtype=np.float32)
    best_idx = np.full(grid_res * grid_res, -1, dtype=np.int32)
    # Track max-y point per bin.
    for i in range(points.shape[0]):
        f = int(flat[i])
        if y[i] > best_y[f]:
            best_y[f] = y[i]
            best_idx[f] = i

    sel = best_idx[best_idx >= 0]
    if sel.size == 0:
        return points[:1]
    return points[sel]

def _downsample(points: np.ndarray, max_points: int, rng: np.random.RandomState) -> np.ndarray:
    if max_points is None or max_points <= 0 or points.shape[0] <= max_points:
        return points
    idx = rng.choice(points.shape[0], size=int(max_points), replace=False)
    return points[idx]

def align_clouds_to_reference_front(
    clouds: List[np.ndarray],
    *,
    params: AlignmentParams,
    reference_idx: int = 0,
    rng_seed: int = 0,
) -> List[np.ndarray]:
    """Align multiple point clouds to a reference using robust yaw search.

    This function aligns *across seeds* (same input view). It is intentionally
    conservative: it only searches small yaw perturbations around the original
    orientation so we don't accidentally "solve" the uncertainty by rotating a
    hallucinated handle into agreement.

    Returns aligned clouds in the reference cloud's original coordinate system.
    """
    if not clouds:
        return []

    ref_i = int(np.clip(reference_idx, 0, len(clouds) - 1))
    rng = np.random.RandomState(int(rng_seed))

    ref = np.asarray(clouds[ref_i], dtype=np.float32)
    if ref.ndim != 2 or ref.shape[1] != 3:
        raise ValueError(f"Expected reference point cloud shape (N,3), got {ref.shape}")

    c0 = _robust_center(ref)
    s0 = _robust_scale(ref, c0)
    ref_n = (ref - c0[None, :]) / float(s0)
    ref_surf = _extract_front_surface(ref_n, params.surface_grid)
    ref_surf = _keep_core_points(ref_surf, params.core_ratio)
    ref_surf = _downsample(ref_surf, params.max_points, rng)

    ref_tree = cKDTree(ref_surf)
    ref_center_surf = _robust_center(ref_surf)

    # Precompute yaw candidates (inclusive range), biased toward small rotations.
    max_yaw = float(abs(params.max_yaw_deg))
    step = float(max(params.yaw_step_deg, 1e-3))
    n = int(math.floor(max_yaw / step + 1e-6))
    # Order: 0, +step, -step, +2step, -2step, ...
    yaws_list: List[float] = [0.0]
    for k in range(1, n + 1):
        yaws_list.append(float(k * step))
        yaws_list.append(float(-k * step))
    # Also include endpoints if max_yaw isn't an integer multiple of step.
    if n == 0 or abs(yaws_list[-2]) < max_yaw - 1e-6:
        yaws_list.append(float(max_yaw))
        yaws_list.append(float(-max_yaw))
    yaws = np.array(yaws_list, dtype=np.float32)

    aligned: List[np.ndarray] = []
    for idx, cloud in enumerate(clouds):
        pts = np.asarray(cloud, dtype=np.float32)
        if pts.size == 0:
            aligned.append(pts)
            continue
        if pts.ndim != 2 or pts.shape[1] != 3:
            raise ValueError(f"Expected point cloud shape (N,3), got {pts.shape}")

        if idx == ref_i:
            aligned.append(ref)
            continue

        ci = _robust_center(pts)
        si = _robust_scale(pts, ci)
        pts_n = (pts - ci[None, :]) / float(si)

        # Extract a stable approximation of the visible/front surface once, then
        # search for a small yaw that best matches the reference surface.
        src_surf0 = _extract_front_surface(pts_n, params.surface_grid)
        src_surf0 = _keep_core_points(src_surf0, params.core_ratio)
        src_surf0 = _downsample(src_surf0, params.max_points, rng)
        src_center0 = _robust_center(src_surf0)

        best_yaw = 0.0
        best_score = float("inf")
        best_t = np.zeros(3, dtype=np.float32)
        yaw_reg = 0.002  # penalize large yaw unless strongly supported

        def _eval_yaw(yaw_deg: float) -> Tuple[float, np.ndarray]:
            R = _rotation_z(float(yaw_deg))
            src_rot = (src_surf0 @ R.T).astype(np.float32)
            t = (ref_center_surf - (src_center0 @ R.T)).astype(np.float32)
            src_rot_t = src_rot + t[None, :]
            # Symmetric NN distance (prevents the score from cheating by matching
            # only a small subset of points).
            d1, _ = ref_tree.query(src_rot_t, k=1, workers=-1)
            src_tree = cKDTree(src_rot_t)
            d2, _ = src_tree.query(ref_surf, k=1, workers=-1)
            score_data = float(np.quantile(d1, 0.9) + np.quantile(d2, 0.9))
            score = score_data + yaw_reg * (abs(float(yaw_deg)) / max(max_yaw, 1e-6)) ** 2
            return score, t

        for yaw in yaws:
            score, t = _eval_yaw(float(yaw))
            if score < best_score:
                best_score = score
                best_yaw = float(yaw)
                best_t = t

        # Fine search around the best coarse yaw for better precision.
        fine_step = max(1.0, step / 5.0)
        fine_lo = max(-max_yaw, best_yaw - step)
        fine_hi = min(max_yaw, best_yaw + step)
        fine_yaws = np.arange(fine_lo, fine_hi + 0.5 * fine_step, fine_step, dtype=np.float32)
        for yaw in fine_yaws:
            score, t = _eval_yaw(float(yaw))
            if score < best_score:
                best_score = score
                best_yaw = float(yaw)
                best_t = t

        R_best = _rotation_z(best_yaw)

        # Optional translation refinement using trimmed correspondences after yaw.
        src_surf_all = (src_surf0 @ R_best.T).astype(np.float32)
        src_surf_t = src_surf_all + best_t[None, :]
        d_all, nn = ref_tree.query(src_surf_t, k=1, workers=-1)
        d_all = np.asarray(d_all, dtype=np.float32)
        nn = np.asarray(nn, dtype=np.int32)
        k = max(1, int(math.ceil(float(params.trim_ratio) * d_all.size)))
        inlier_idx = np.argpartition(d_all, k - 1)[:k]
        if inlier_idx.size > 0:
            dst_in = ref_surf[nn[inlier_idx]]
            src_in = src_surf_all[inlier_idx]
            # Solve for translation only (rotation fixed) in normalized space.
            best_t = (dst_in.mean(axis=0) - src_in.mean(axis=0)).astype(np.float32)

        pts_aligned_n = (pts_n @ R_best.T).astype(np.float32) + best_t[None, :]
        pts_aligned = pts_aligned_n * float(s0) + c0[None, :]
        aligned.append(pts_aligned.astype(np.float32))

    return aligned
