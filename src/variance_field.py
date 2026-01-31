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
    """Compute occupancy variance across K point clouds on a voxel grid.
    
    We apply Gaussian blurring to individual occupancy grids before computing
    variance. This allows points that are 'close' across different seeds to
    overlap, reducing noise and concentrating variance on truly ambiguous regions.
    """
    grid = VoxelGrid(bounds=cfg.grid_bounds, resolution=cfg.resolution)
    occupancy = []
    
    for pc in point_clouds:
        occ = _occupancy_grid_from_points(pc, grid)
        occ_f = occ.astype(np.float32)
        
        # Apply blurring to each seed's occupancy to handle point jitter
        if cfg.gaussian_sigma and gaussian_filter is not None:
            # IMPORTANT: use constant padding to avoid boundary reflection artifacts.
            # SciPy's default mode='reflect' can create a visible "box"/frame in
            # max-projection figures when occupancy touches grid borders.
            occ_f = gaussian_filter(
                occ_f,
                sigma=cfg.gaussian_sigma,
                mode="constant",
                cval=0.0,
            )
            
        occupancy.append(occ_f)
        
    if not occupancy:
        return np.zeros((grid.resolution, grid.resolution, grid.resolution), dtype=np.float32)

    stack = np.stack(occupancy, axis=0)
    mean = stack.mean(axis=0)
    
    # Bernoulli variance: p(1-p). 
    # With blurred p, this is low where seeds consistently agree on occupancy.
    var = mean * (1.0 - mean)  

    return var


def accumulate_semantic_weights(
    point_clouds: List[np.ndarray],
    point_weights: List[np.ndarray],
    cfg: VarianceConfig,
) -> np.ndarray:
    """Aggregate per-point semantic weights into a voxel grid stack (N, Res, Res, Res)."""
    grid = VoxelGrid(bounds=cfg.grid_bounds, resolution=cfg.resolution)
    
    semantic_stack = []
    
    for pc, w in zip(point_clouds, point_weights):
        # Create grid for this seed
        seed_accum = np.zeros((grid.resolution, grid.resolution, grid.resolution), dtype=np.float32)
        seed_counts = np.zeros_like(seed_accum)
        
        idx, valid = grid.world_to_grid(pc)
        idx = idx[valid]
        w = w[valid]
        
        if idx.size > 0:
            np.add.at(seed_accum, (idx[:, 0], idx[:, 1], idx[:, 2]), w)
            np.add.at(seed_counts, (idx[:, 0], idx[:, 1], idx[:, 2]), 1.0)
            
        mask = seed_counts > 0
        seed_grid = np.zeros_like(seed_accum)
        seed_grid[mask] = seed_accum[mask] / seed_counts[mask]
        
        semantic_stack.append(seed_grid)
        
    if not semantic_stack:
        return np.zeros((0, grid.resolution, grid.resolution, grid.resolution), dtype=np.float32)

    return np.stack(semantic_stack, axis=0)


def combine_variance_and_semantics(
    variance: np.ndarray,
    semantic_stack: np.ndarray,
    cfg: VarianceConfig,
    *,
    return_components: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict]:
    """Combine geometric variance with semantic weights using primary and secondary signals.

    If `return_components=True`, returns `(scores, components)` where `components` includes
    intermediate grids useful for debugging why handle/backside regions are not being
    highlighted:
    - `mean_semantic`: mean semantic weight per voxel
    - `semantic_var_raw`: raw semantic variance per voxel (before gating/power)
    - `s1`: primary signal (Var(Geometry) * gated Mean(Semantics))
    - `s2`: secondary signal (gated Var(Semantics) after power)
    """
    # semantic_stack is (N, V, V, V)
    #
    # Design note (Active Hallucination):
    # - `mean_semantic` captures *consensus* about where the affordance is.
    # - `semantic_var_raw` captures *disagreement* across hallucinated hypotheses.
    #
    # For the secondary (disagreement) signal, gating by the mean can accidentally
    # erase the "one-of-K" case we care about (e.g., only 1 seed hallucinates the
    # handle at voxel v => mean is small but uncertainty is real). To better match
    # the intended behavior, we gate S2 by `max_semantic` ("any hypothesis thinks
    # this voxel is affordance-like") rather than the mean.
    if semantic_stack.shape[0] > 0:
        mean_semantic = semantic_stack.mean(axis=0)
        max_semantic = semantic_stack.max(axis=0)
        semantic_var_raw = np.var(semantic_stack, axis=0)
    else:
        mean_semantic = np.zeros_like(variance)
        max_semantic = np.zeros_like(variance)
        semantic_var_raw = np.zeros_like(variance)

    # Focus S1 on high-confidence semantic regions.
    # This helps prevent "square"/bounding-box artifacts from dominating when
    # CLIPSeg produces diffuse low-confidence masks.
    sem_s1 = mean_semantic
    thr1 = float(getattr(cfg, "semantic_threshold", 0.0) or 0.0)
    if thr1 > 0.0:
        sem_s1 = np.clip((sem_s1 - thr1) / max(1.0 - thr1, 1e-6), 0.0, 1.0)
    gamma1 = float(getattr(cfg, "semantic_gamma", 1.0) or 1.0)
    if abs(gamma1 - 1.0) > 1e-6:
        sem_s1 = np.power(np.clip(sem_s1, 0.0, 1.0), gamma1)

    s1 = variance * sem_s1

    # Secondary signal: semantic disagreement across seeds.
    # Use a *softer* semantic gate than S1 so we don't erase disagreements when
    # only a subset of seeds predict the affordance (low mean, high variance).
    # IMPORTANT: gate by max_semantic ("exists in any hypothesis"), not mean.
    sem_s2 = max_semantic
    thr2 = float(getattr(cfg, "semantic_s2_threshold", 0.0) or 0.0)
    if thr2 > 0.0:
        sem_s2 = np.clip((sem_s2 - thr2) / max(1.0 - thr2, 1e-6), 0.0, 1.0)
    gamma2 = float(getattr(cfg, "semantic_s2_gamma", 1.0) or 1.0)
    if abs(gamma2 - 1.0) > 1e-6:
        sem_s2 = np.power(np.clip(sem_s2, 0.0, 1.0), gamma2)

    s2 = semantic_var_raw * sem_s2
    p2 = float(getattr(cfg, "semantic_s2_power", 1.0) or 1.0)
    # Power < 1 boosts small values (e.g., sqrt).
    if abs(p2 - 1.0) > 1e-6:
        s2 = np.power(np.clip(s2, 0.0, None), p2)

    # Combined Score
    scores = s1 + cfg.semantic_variance_weight * s2
    
    min_val = scores.min()
    max_val = scores.max()
    
    if max_val - min_val > 1e-8:
        scores = (scores - min_val) / (max_val - min_val)
    else:
        scores = np.zeros_like(scores)
        
    if return_components:
        comps = {
            "mean_semantic": np.asarray(mean_semantic, dtype=np.float32),
            "max_semantic": np.asarray(max_semantic, dtype=np.float32),
            "semantic_var_raw": np.asarray(semantic_var_raw, dtype=np.float32),
            "s1": np.asarray(s1, dtype=np.float32),
            "s2": np.asarray(s2, dtype=np.float32),
        }
        return np.asarray(scores, dtype=np.float32), comps

    return np.asarray(scores, dtype=np.float32)


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


def export_topk_score_points(score_grid: np.ndarray, grid: VoxelGrid, topk_ratio: float) -> np.ndarray:
    """Return (N,4) array of top-k voxels as [x,y,z,score] in world coordinates.

    This is similar to [`extract_topk_centroid()`](src/variance_field.py:178) but also
    returns each selected voxel's score so callers can do weighted view scoring.
    """
    flat = score_grid.ravel()
    k = max(1, int(topk_ratio * flat.shape[0]))

    if flat.size == 0:
        return np.zeros((0, 4), dtype=np.float32)

    if k < flat.shape[0]:
        topk_idx = np.argpartition(-flat, k)[:k]
    else:
        topk_idx = np.arange(flat.shape[0])

    coords_idx = np.column_stack(np.unravel_index(topk_idx, score_grid.shape))
    centers = grid.grid_to_world(coords_idx).astype(np.float32)
    scores = flat[topk_idx].astype(np.float32)
    if centers.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    return np.concatenate([centers, scores[:, None]], axis=1)


def export_score_points(score_grid: np.ndarray, grid: VoxelGrid, threshold: float = 0.1) -> np.ndarray:
    """Convert voxels above threshold into point coordinates with scores as attributes."""
    mask = score_grid >= threshold
    idx = np.column_stack(np.nonzero(mask))
    
    if idx.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
        
    centers = grid.grid_to_world(idx)
    scores = score_grid[mask]
    
    return np.concatenate([centers, scores[:, None]], axis=1)
