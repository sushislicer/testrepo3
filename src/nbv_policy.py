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
        score = align ** cfg.alignment_pow
        scores.append((score, i))
    if not scores:
        # fallback to any unvisited view
        remaining = [i for i in range(len(poses)) if i not in visited]
        if not remaining:
            return current_idx, 0.0
        return remaining[0], 0.0
    scores.sort(key=lambda x: x[0], reverse=True)
    return scores[0][1], scores[0][0]


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
