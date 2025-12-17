"""Unit tests for NBV policies."""

import unittest
import sys
import random
import math
import numpy as np
from dataclasses import dataclass
from pathlib import Path

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

@dataclass
class Pose:
    position: np.ndarray
    look_at: np.ndarray
    id: int = 0

@dataclass
class PolicyConfig:
    min_angle_deg: float = 0.0
    alignment_pow: float = 1.0

try:
    from src.nbv_policy import (
        select_next_view_active,
        select_next_view_geometric,
        select_next_view_random,
        _view_direction
    )
except ImportError:
    def _view_direction(pose: Pose) -> np.ndarray:
        vec = pose.look_at - pose.position
        return vec / (np.linalg.norm(vec) + 1e-8)

    def select_next_view_random(current_idx, poses, visited, rng):
        remaining = [i for i in range(len(poses)) if i not in visited]
        return rng.choice(remaining) if remaining else current_idx

    def select_next_view_geometric(current_idx, poses, visited):
        cur_dir = _view_direction(poses[current_idx])
        best_idx, best_angle = current_idx, -1.0
        for i, pose in enumerate(poses):
            if i in visited: continue
            dir_vec = _view_direction(pose)
            cosang = float(np.clip(np.dot(cur_dir, dir_vec), -1.0, 1.0))
            angle = math.degrees(math.acos(cosang))
            if angle > best_angle: best_angle, best_idx = angle, i
        return best_idx

    def select_next_view_active(current_idx, poses, target_centroid, visited, cfg):
        scores = []
        for i, pose in enumerate(poses):
            if i in visited: continue
            direction = _view_direction(pose)
            to_target = target_centroid - pose.position
            align = np.dot(direction, to_target / (np.linalg.norm(to_target) + 1e-8))
            score = max(align, 0.0) ** cfg.alignment_pow
            scores.append((score, i))
        if not scores: return current_idx, 0.0
        scores.sort(key=lambda x: x[0], reverse=True)
        return scores[0][1], scores[0][0]


class TestNBVPolicies(unittest.TestCase):

    def setUp(self):
        self.center = np.array([0.0, 0.0, 0.0])
        self.poses = [
            Pose(np.array([2.0, 0.0, 0.0]), self.center, id=0),  # +X (Front)
            Pose(np.array([0.0, 2.0, 0.0]), self.center, id=1),  # +Y (Left)
            Pose(np.array([-2.0, 0.0, 0.0]), self.center, id=2), # -X (Back)
            Pose(np.array([0.0, -2.0, 0.0]), self.center, id=3), # -Y (Right)
        ]
        self.cfg = PolicyConfig()

    def test_view_direction_vector(self):
        direction = _view_direction(self.poses[0])
        expected = np.array([-1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(direction, expected)

    def test_baseline_random(self):
        visited = {0, 1, 3}
        rng = random.Random(42)
        next_view = select_next_view_random(0, self.poses, visited, rng)
        self.assertEqual(next_view, 2)

    def test_baseline_geometric(self):
        # Should pick farthest unvisited view (180 deg)
        visited = {0}
        next_view = select_next_view_geometric(0, self.poses, visited)
        self.assertEqual(next_view, 2)
        
        # Should pick 90 deg neighbor if 180 is visited
        visited = {0, 2}
        next_view = select_next_view_geometric(0, self.poses, visited)
        self.assertIn(next_view, [1, 3])

    def test_active_hallucination_alignment(self):
        # Target at -0.5 (back); Camera 2 (-2.0) has best view
        target_centroid = np.array([-0.5, 0.0, 0.0])
        visited = {0}
        next_view, _ = select_next_view_active(0, self.poses, target_centroid, visited, self.cfg)
        self.assertEqual(next_view, 2)

    def test_active_hallucination_sharpening(self):
        target = np.array([0.0, -0.1, 0.0])
        visited = set()
        
        self.cfg.alignment_pow = 1.0
        _, score_linear = select_next_view_active(0, self.poses, target, visited, self.cfg)
        
        self.cfg.alignment_pow = 5.0
        _, score_sharp = select_next_view_active(0, self.poses, target, visited, self.cfg)
        
        self.assertNotEqual(score_linear, score_sharp)


if __name__ == '__main__':
    unittest.main()