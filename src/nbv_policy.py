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


def _view_position_dir(pose: Pose) -> np.ndarray:
    """Unit vector from object center (origin) to camera position.

    Our orbital poses always look at (approximately) the object center.
    For NBV on a centered object, the most discriminative quantity is often the
    *camera position direction* (which side of the object we are on), not the
    optical axis alignment to a near-center target.
    """
    p = np.asarray(pose.position, dtype=np.float32)
    return p / (np.linalg.norm(p) + 1e-8)


def select_next_view_active(
    current_idx: int,
    poses: List[Pose],
    target_centroid: np.ndarray,
    visited: Set[int],
    cfg: PolicyConfig,
) -> Tuple[int, float]:
    """Select view that best *faces* the target centroid.

    NOTE:
    The camera poses in this project are orbital and look at the object center.
    If a target lies near the center (common when score-fields form rings/shells),
    optical-axis alignment `dot(view_dir, to_target_dir)` becomes ~1 for *all* views
    and the policy degenerates.

    Instead, we score a candidate view by how well its **camera position direction**
    aligns with the target direction from the object center.
    """
    scores = []

    cur_dir = _view_direction(poses[current_idx])

    c = np.asarray(target_centroid, dtype=np.float32).reshape(3)
    c_norm = float(np.linalg.norm(c))
    if c_norm < 1e-6:
        # Degenerate: target near center. Fall back to geometric diversity.
        next_view = select_next_view_geometric(current_idx, poses, visited)
        return int(next_view), 0.0
    c_hat = c / (c_norm + 1e-8)

    for i, pose in enumerate(poses):
        if i in visited:
            continue

        # Enforce movement diversity relative to current view.
        direction = _view_direction(pose)
        align_move = float(np.clip(np.dot(cur_dir, direction), -1.0, 1.0))
        angle_move = math.degrees(math.acos(align_move))
        if float(getattr(cfg, "min_angle_deg", 0.0) or 0.0) > 0.0 and angle_move < float(cfg.min_angle_deg):
            continue

        # Facing score: does the camera sit on the same side as the target?
        p_hat = _view_position_dir(pose)
        face = float(np.dot(p_hat, c_hat))
        face = float(np.clip(face, 0.0, 1.0))
        base = face ** float(cfg.alignment_pow)

        # Prefer larger viewpoint changes (softly) to avoid tiny azimuth increments.
        div_w = float(getattr(cfg, "diversity_weight", 0.0) or 0.0)
        div_p = float(getattr(cfg, "diversity_pow", 1.0) or 1.0)
        move = float(np.clip(angle_move / 180.0, 0.0, 1.0))
        move = move ** max(div_p, 1e-6)
        score = base * (1.0 + div_w * move)
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

    # Use the *direction* of the weighted centroid as a coarse facing reference.
    weighted_centroid = (xyz * w[:, None]).sum(axis=0)
    c_norm = float(np.linalg.norm(weighted_centroid))
    c_hat = weighted_centroid / (c_norm + 1e-8) if c_norm > 1e-6 else None

    scores: List[Tuple[float, int]] = []
    pow_ = float(cfg.alignment_pow)
    min_angle = float(getattr(cfg, "min_angle_deg", 0.0) or 0.0)

    cur_dir = _view_direction(poses[current_idx])

    for i, pose in enumerate(poses):
        if i in visited:
            continue

        # Enforce movement diversity relative to current view.
        direction = _view_direction(pose)
        align_move = float(np.clip(np.dot(cur_dir, direction), -1.0, 1.0))
        angle_move = math.degrees(math.acos(align_move))
        if min_angle > 0.0 and angle_move < min_angle:
            continue

        # Facing score: expected visibility of score-weighted targets.
        # A surface voxel at xyz is most visible when the camera position points
        # toward xyz (same-side hemisphere). Using camera *position direction*
        # avoids the "all views look at origin" degeneracy.
        p_hat = _view_position_dir(pose)

        r = np.linalg.norm(xyz, axis=1)
        valid = r > 1e-6
        if not np.any(valid):
            # No meaningful spatial direction; fall back to centroid direction if available.
            if c_hat is None:
                continue
            face = float(np.dot(p_hat, c_hat))
            face = float(np.clip(face, 0.0, 1.0))
            score = face ** pow_
        else:
            xyz_hat = np.zeros_like(xyz)
            xyz_hat[valid] = xyz[valid] / r[valid, None]
            face = np.sum(xyz_hat * p_hat[None, :], axis=1)
            face = np.clip(face, 0.0, 1.0)
            base = float(np.sum(w * (face ** pow_)))

            div_w = float(getattr(cfg, "diversity_weight", 0.0) or 0.0)
            div_p = float(getattr(cfg, "diversity_pow", 1.0) or 1.0)
            move = float(np.clip(angle_move / 180.0, 0.0, 1.0))
            move = move ** max(div_p, 1e-6)
            score = base * (1.0 + div_w * move)
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
