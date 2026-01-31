"""Geometry-only affordance proxies.

This module provides *weak* fallbacks when 2D semantic segmentation (CLIPSeg) is
unreliable on point-cloud renders.

Motivation:
- When the handle is occluded in the conditioning view, CLIPSeg on sparse
  point-cloud renders can return near-zero masks for all Point-E seeds.
- For mugs, the handle typically manifests as a radial protrusion from the body.
  We can detect this protrusion from geometry alone and use it to:
  (1) rank oversampled Point-E seeds (keep the most handle-like hypotheses)
  (2) provide a non-zero "semantic" stack so the combined score is not degenerate.
"""

from __future__ import annotations

import numpy as np


def protrusion_weights(
    points: np.ndarray,
    *,
    core_quantile: float = 0.85,
    mad_k: float = 3.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Return per-point weights (N,) emphasizing radial protrusions.

    The returned weights are in [0,1]. For mug-like shapes, handle points tend to
    lie at larger radius than the main body, so they get higher weights.

    Returns:
      (weights, center) where center is the robust median center (3,).
    """
    pts = np.asarray(points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((3,), dtype=np.float32)

    center = np.median(pts, axis=0).astype(np.float32)
    d = np.linalg.norm(pts - center[None, :], axis=1).astype(np.float32)
    if d.size == 0:
        return np.zeros((0,), dtype=np.float32), center

    core_q = float(np.clip(core_quantile, 0.5, 0.99))
    q = float(np.quantile(d, core_q))
    med = float(np.median(d))
    mad = float(np.median(np.abs(d - med))) + 1e-6
    thr = float(max(q, med + float(mad_k) * mad))

    d_max = float(d.max())
    denom = max(d_max - thr, 1e-6)
    w = np.clip((d - thr) / denom, 0.0, 1.0).astype(np.float32)
    return w, center


def protrusion_direction(points: np.ndarray, weights: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Weighted direction (3,) from the center toward protrusion points."""
    pts = np.asarray(points, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    c = np.asarray(center, dtype=np.float32).reshape(3)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.size == 0 or w.size != pts.shape[0]:
        return np.zeros((3,), dtype=np.float32)
    s = float(np.sum(w))
    if s <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    v = np.sum((pts - c[None, :]) * w[:, None], axis=0)
    n = float(np.linalg.norm(v))
    if n <= 1e-8:
        return np.zeros((3,), dtype=np.float32)
    return (v / n).astype(np.float32)


def protrusion_strength(weights: np.ndarray, *, q: float = 0.995) -> float:
    """Scalar strength score in [0,1] based on high-quantile protrusion weight."""
    w = np.asarray(weights, dtype=np.float32).reshape(-1)
    if w.size == 0:
        return 0.0
    qq = float(np.clip(q, 0.5, 1.0))
    return float(np.quantile(w, qq))

