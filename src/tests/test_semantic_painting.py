"""
Unit tests for multiview semantic painting helpers.
Usage: python src/tests/test_semantic_painting.py
"""

import unittest
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.segmentation import (
    compute_point_mask_weights_multiview,
    project_points_to_pixels,
    render_point_cloud_view,
    rotate_cam_to_world_yaw,
)
from src.simulator import Pose, pose_to_matrix


@dataclass
class SimpleIntrinsics:
    width: int = 100
    height: int = 100
    fx: float = 100.0
    fy: float = 100.0
    cx: float = 50.0
    cy: float = 50.0


class _CountingSegmenter:
    """Deterministic stub: returns ones except for specified call indices (1-based)."""

    def __init__(self, zero_calls: set[int]):
        self.zero_calls = set(zero_calls)
        self.calls = 0

    def segment_affordance(self, image: np.ndarray, prompt=None) -> np.ndarray:
        self.calls += 1
        h, w = image.shape[:2]
        if self.calls in self.zero_calls:
            return np.zeros((h, w), dtype=np.float32)
        return np.ones((h, w), dtype=np.float32)


class TestSemanticPainting(unittest.TestCase):

    def setUp(self):
        self.intr = SimpleIntrinsics()
        self.pose = Pose(
            position=np.array([1.0, 0.0, 0.5], dtype=np.float32),
            look_at=np.zeros(3, dtype=np.float32),
            up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
        )
        self.cam_to_world = pose_to_matrix(self.pose)
        self.points = np.array(
            [
                [0.0, 0.0, 0.0],
                [0.05, 0.0, 0.0],
                [0.0, 0.05, 0.0],
            ],
            dtype=np.float32,
        )

    def test_project_points_center(self):
        pix = project_points_to_pixels(self.points[:1], self.cam_to_world, self.intr)
        u, v, depth = float(pix[0, 0]), float(pix[0, 1]), float(pix[0, 2])
        self.assertGreater(depth, 0.0)
        self.assertAlmostEqual(u, self.intr.cx, places=3)
        self.assertAlmostEqual(v, self.intr.cy, places=3)

    def test_rotate_cam_to_world_yaw(self):
        rotated = rotate_cam_to_world_yaw(self.cam_to_world, 90.0)
        pos = rotated[:3, 3]
        # (x, y) should rotate: (1,0) -> (0,1) up to small numerical error.
        self.assertAlmostEqual(float(pos[0]), 0.0, places=4)
        self.assertAlmostEqual(float(pos[1]), 1.0, places=4)

    def test_render_point_cloud_view_nonempty(self):
        img = render_point_cloud_view(self.points, self.cam_to_world, self.intr, point_radius_px=1, blur_sigma=0.0)
        self.assertEqual(img.shape, (self.intr.height, self.intr.width, 3))
        self.assertGreater(int(img.sum()), 0)

    def test_multiview_weight_aggregation(self):
        # View-2 returns zeros, views 1 and 3 return ones -> mean = 2/3 for all visible points.
        segmenter = _CountingSegmenter(zero_calls={2})
        weights = compute_point_mask_weights_multiview(
            self.points,
            self.cam_to_world,
            self.intr,
            segmenter,
            num_views=3,
            yaw_step_deg=120.0,
            aggregation="mean",
            point_radius_px=1,
            blur_sigma=0.0,
        )
        self.assertEqual(weights.shape, (self.points.shape[0],))
        self.assertTrue(np.all(weights >= 0.0))
        self.assertTrue(np.all(weights <= 1.0))
        self.assertTrue(np.allclose(weights, 2.0 / 3.0, atol=1e-4))


if __name__ == "__main__":
    unittest.main()

