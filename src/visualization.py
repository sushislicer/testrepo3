"""Lightweight visualization utilities for figures and debugging."""

from __future__ import annotations

import colorsys
from pathlib import Path
from typing import List

import numpy as np
import matplotlib.pyplot as plt

try:
    import open3d as o3d
except Exception:  # pragma: no cover - optional dependency
    o3d = None


def overlay_point_clouds_open3d(point_clouds: List[np.ndarray]) -> None:
    """Quick visual overlay using open3d (interactive)."""
    if o3d is None:
        raise ImportError("open3d is required for visualization.")
    geoms = []
    for idx, pc in enumerate(point_clouds):
        color = np.array(colorsys.hsv_to_rgb(idx / max(len(point_clouds), 1), 0.7, 1.0))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (pc.shape[0], 1)))
        geoms.append(pcd)
    o3d.visualization.draw_geometries(geoms)


def save_variance_max_projection(score_grid: np.ndarray, path: str) -> None:
    """Save top-down max projection of variance/score grid."""
    proj = np.max(score_grid, axis=2)
    proj = proj / (proj.max() + 1e-8)
    plt.figure(figsize=(4, 4))
    plt.imshow(proj, cmap="inferno")
    plt.axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()


def save_trajectory_grid(images: List[np.ndarray], path: str) -> None:
    """Save a horizontal strip of trajectory RGB images."""
    if not images:
        return
    h, w, _ = images[0].shape
    canvas = np.zeros((h, w * len(images), 3), dtype=np.uint8)
    for i, img in enumerate(images):
        canvas[:, i * w : (i + 1) * w] = img
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    import imageio

    imageio.imwrite(path, canvas)
