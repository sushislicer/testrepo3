"""Voxel grid utilities for semantic variance computation."""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple

try:
    from scipy.ndimage import gaussian_filter
except ImportError:  # pragma: no cover
    gaussian_filter = None

from .config import VarianceConfig

@dataclass
class VoxelGrid:
    bounds: float
    resolution: int

    @property
    def voxel_size(self) -> float:
        return 2 * self.bounds / self.resolution

    def world_to_grid(self, points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Map world points (N,3) to voxel indices (N,3) and validity mask."""
        coords = (points + self.bounds) / (2 * self.bounds)
        idx = np.floor(coords * self.resolution).astype(int)
        valid = (idx >= 0).all(axis=1) & (idx < self.resolution).all(axis=1)
        return idx, valid

    def grid_to_world(self, idx: np.ndarray) -> np.ndarray:
        """Convert voxel indices (N,3) to world-space centers."""
        centers = (idx + 0.5) * self.voxel_size - self.bounds
        return centers


def _occupancy_grid_from_points(points: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    idx, valid = grid.world_to_grid(points)
    idx = idx[valid]
    occ = np.zeros((grid.resolution, grid.resolution, grid.resolution), dtype=bool)
    if idx.size > 0:
        occ[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return occ


def compute_variance_field(point_clouds: List[np.ndarray], cfg: VarianceConfig) -> np.ndarray:
    """Compute occupancy variance across K point clouds on a voxel grid."""
    grid = VoxelGrid(bounds=cfg.grid_bounds, resolution=cfg.resolution)
    occupancy = []
    
    for pc in point_clouds:
        occ = _occupancy_grid_from_points(pc, grid)
        occupancy.append(occ.astype(np.float32))
        
    if not occupancy:
        return np.zeros((grid.resolution, grid.resolution, grid.resolution), dtype=np.float32)

    stack = np.stack(occupancy, axis=0)
    mean = stack.mean(axis=0)
    # Bernoulli variance: p(1-p)
    var = mean * (1.0 - mean)  

    if cfg.gaussian_sigma and gaussian_filter is not None:
        var = gaussian_filter(var, sigma=cfg.gaussian_sigma)
        
    return var


def accumulate_semantic_weights(
    point_clouds: List[np.ndarray],
    point_weights: List[np.ndarray],
    cfg: VarianceConfig,
) -> np.ndarray:
    """Aggregate per-point semantic weights into a voxel grid using vectorized ops."""
    grid = VoxelGrid(bounds=cfg.grid_bounds, resolution=cfg.resolution)
    accum = np.zeros((grid.resolution, grid.resolution, grid.resolution), dtype=np.float32)
    counts = np.zeros_like(accum)

    for pc, w in zip(point_clouds, point_weights):
        idx, valid = grid.world_to_grid(pc)
        idx = idx[valid]
        w = w[valid]
        
        if idx.size > 0:
            # Vectorized accumulation for speed (removes Python loop)
            np.add.at(accum, (idx[:, 0], idx[:, 1], idx[:, 2]), w)
            np.add.at(counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
            
    # Avoid division by zero
    mask = counts > 0
    semantic = np.zeros_like(accum)
    semantic[mask] = accum[mask] / counts[mask]
    
    return semantic


def combine_variance_and_semantics(variance: np.ndarray, semantic: np.ndarray) -> np.ndarray:
    """Element-wise multiplication with Min-Max normalization to [0,1]."""
    scores = variance * semantic
    
    min_val = scores.min()
    max_val = scores.max()
    
    if max_val - min_val > 1e-8:
        scores = (scores - min_val) / (max_val - min_val)
    else:
        scores = np.zeros_like(scores)
        
    return scores


def extract_topk_centroid(score_grid: np.ndarray, grid: VoxelGrid, topk_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """Return centroid (3,) of top-k scoring voxels and their world coords."""
    flat = score_grid.ravel()
    k = max(1, int(topk_ratio * flat.shape[0]))
    
    if k < flat.shape[0]:
        topk_idx = np.argpartition(-flat, k)[:k]
    else:
        topk_idx = np.arange(flat.shape[0])

    coords_idx = np.column_stack(np.unravel_index(topk_idx, score_grid.shape))
    centers = grid.grid_to_world(coords_idx)
    
    if centers.size > 0:
        centroid = centers.mean(axis=0)
    else:
        centroid = np.zeros(3, dtype=np.float32)
        
    return centroid, centers


def export_score_points(score_grid: np.ndarray, grid: VoxelGrid, threshold: float = 0.1) -> np.ndarray:
    """Convert voxels above threshold into point coordinates with scores as attributes."""
    mask = score_grid >= threshold
    idx = np.column_stack(np.nonzero(mask))
    
    if idx.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
        
    centers = grid.grid_to_world(idx)
    scores = score_grid[mask]
    
    return np.concatenate([centers, scores[:, None]], axis=1)