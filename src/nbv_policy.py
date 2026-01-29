"""Next-Best-View policies: Active-Hallucination, Random, Geometric."""

from __future__ import annotations

import math
import random
from typing import Iterable, List, Tuple, Set

import numpy as np

from .config import PolicyConfig
from .simulator import Pose


def _view_direction(pose: Pose) -> np.ndarray:
    vec = pose.look_at - pose.position
    norm = np.linalg.norm(vec) + 1e-8
    return vec / norm


def select_next_view_active(
    current_idx: int,
    poses: List[Pose],
    target_centroid: np.ndarray,
    visited: Set[int],
    cfg: PolicyConfig,
) -> Tuple[int, float]:
    """Select view whose optical axis best aligns with the target centroid."""
    scores = []
    for i, pose in enumerate(poses):
        if i in visited:
            continue
        direction = _view_direction(pose)
        to_target = target_centroid - pose.position
        if np.linalg.norm(to_target) < 1e-6:
            align = 1.0
        else:
            align = np.dot(direction, to_target / (np.linalg.norm(to_target) + 1e-8))
        align = max(align, -1.0)
        angle = math.degrees(math.acos(max(min(align, 1.0), -1.0)))
        if angle < cfg.min_angle_deg:
            continue
        # Use only non-negative alignment for scoring so that raising to a
        # fractional power never produces complex values when the target lies
        # behind the camera.
        align_for_score = max(align, 0.0)
        score = align_for_score ** cfg.alignment_pow
        scores.append((score, i))
    if not scores:
        # fallback to any unvisited view
        remaining = [i for i in range(len(poses)) if i not in visited]
        if not remaining:
            return current_idx, 0.0
        return remaining[0], 0.0
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1], scores[0][0]


def select_next_view_active_weighted(
    current_idx: int,
    poses: List[Pose],
    score_points: np.ndarray,
    visited: Set[int],
    cfg: PolicyConfig,
) -> Tuple[int, float]:
    """Active NBV using a *distribution* of 3D targets rather than a single centroid.

    `score_points` is expected to be an (N,4) array of [x,y,z,score] in the same
    coordinate frame as `poses`.

    Rationale:
    - For *_combined variants, the 3D score field already encodes semantic signal
      (S1 + w*S2). Collapsing it to a centroid can lose structure (rings/shells).
    - We score a candidate view by its expected alignment toward many high-score
      voxels, weighted by their score.
    """

    pts = np.asarray(score_points, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 4 or pts.shape[0] == 0:
        remaining = [i for i in range(len(poses)) if i not in visited]
        if not remaining:
            return current_idx, 0.0
        return remaining[0], 0.0

    xyz = pts[:, :3]
    w = np.clip(pts[:, 3], 0.0, None)
    w_sum = float(w.sum())
    if w_sum <= 1e-8:
        w = np.ones_like(w)
        w_sum = float(w.sum())
    w = w / w_sum

    # Use a weighted centroid only for the optional min-angle gating.
    weighted_centroid = (xyz * w[:, None]).sum(axis=0)

    scores: List[Tuple[float, int]] = []
    pow_ = float(cfg.alignment_pow)
    min_angle = float(getattr(cfg, "min_angle_deg", 0.0) or 0.0)

    for i, pose in enumerate(poses):
        if i in visited:
            continue

        direction = _view_direction(pose)
        to_centroid = weighted_centroid - pose.position
        if np.linalg.norm(to_centroid) > 1e-6:
            align_c = float(np.dot(direction, to_centroid / (np.linalg.norm(to_centroid) + 1e-8)))
            align_c = float(np.clip(align_c, -1.0, 1.0))
            angle_c = math.degrees(math.acos(align_c))
            if min_angle > 0.0 and angle_c < min_angle:
                continue

        # Score: expected alignment toward score-weighted targets.
        to = xyz - pose.position[None, :]
        d = np.linalg.norm(to, axis=1) + 1e-8
        to_n = to / d[:, None]
        align = np.sum(to_n * direction[None, :], axis=1)
        align = np.clip(align, 0.0, 1.0)
        score = float(np.sum(w * (align ** pow_)))
        scores.append((score, i))

    if not scores:
        remaining = [i for i in range(len(poses)) if i not in visited]
        if not remaining:
            return current_idx, 0.0
        return remaining[0], 0.0

    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1], float(scores[0][0])


def select_next_view_random(current_idx: int, poses: List[Pose], visited: Set[int], rng: random.Random) -> int:
    remaining = [i for i in range(len(poses)) if i not in visited]
    if not remaining:
        return current_idx
    return rng.choice(remaining)


def select_next_view_geometric(current_idx: int, poses: List[Pose], visited: Set[int]) -> int:
    """Choose view with maximum angular distance from current pose."""
    cur_dir = _view_direction(poses[current_idx])
    best_idx = current_idx
    best_angle = -1.0
    for i, pose in enumerate(poses):
        if i in visited:
            continue
        dir_vec = _view_direction(pose)
        cosang = float(np.clip(np.dot(cur_dir, dir_vec), -1.0, 1.0))
        angle = math.degrees(math.acos(cosang))
        if angle > best_angle:
            best_angle = angle
            best_idx = i
    return best_idx
