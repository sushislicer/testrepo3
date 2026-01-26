"""
Unit tests for cloud-frame alignment helpers.
Usage: python -m unittest src.tests.test_cloud_frame
"""

import unittest
from dataclasses import dataclass

import numpy as np

from src.cloud_frame import (
    compute_object_mask,
    estimate_orbit_distance_for_cloud,
    estimate_yaw_align_to_mask,
    make_cloud_orbital_poses,
    normalize_clouds_to_bounds,
    rotate_clouds_yaw,
)
from src.segmentation import render_point_cloud_view
from src.simulator import pose_to_matrix


@dataclass
class SimpleIntrinsics:
    width: int = 128
    height: int = 128
    fx: float = 180.0
    fy: float = 180.0
    cx: float = 64.0
    cy: float = 64.0


def _angle_diff(a: float, b: float) -> float:
    x = abs((a - b) % 360.0)
    return min(x, 360.0 - x)


def _make_mug_like_point_cloud(n_body: int = 1500, n_handle: int = 400) -> np.ndarray:
    # Body: hollow-ish cylinder surface.
    rng = np.random.RandomState(0)
    theta = rng.rand(n_body) * 2 * np.pi
    z = rng.rand(n_body) * 0.7 - 0.1
    r = 0.25 + 0.01 * rng.randn(n_body)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    body = np.stack([x, y, z], axis=1).astype(np.float32)

    # Handle: offset arc on +X side (breaks yaw symmetry).
    t = rng.rand(n_handle) * np.pi
    handle_r = 0.10 + 0.005 * rng.randn(n_handle)
    hx = 0.38 + handle_r * np.cos(t)
    hy = 0.00 + handle_r * np.sin(t)
    hz = 0.25 + 0.06 * rng.randn(n_handle)
    handle = np.stack([hx, hy, hz], axis=1).astype(np.float32)

    pts = np.concatenate([body, handle], axis=0)
    # Center roughly.
    pts = pts - np.median(pts, axis=0, keepdims=True)
    return pts.astype(np.float32)


class TestCloudFrame(unittest.TestCase):
    def setUp(self) -> None:
        self.intr = SimpleIntrinsics()

    def test_normalize_clouds_to_bounds(self):
        cloud = _make_mug_like_point_cloud()
        clouds_norm, norm = normalize_clouds_to_bounds([cloud], grid_bounds=0.5, margin=0.95)
        self.assertEqual(len(clouds_norm), 1)
        pts = clouds_norm[0]
        self.assertLessEqual(float(np.quantile(np.abs(pts), 0.995)), 0.5 * 0.95 + 1e-3)
        self.assertEqual(norm.center.shape, (3,))
        self.assertGreater(norm.scale, 0.0)

    def test_estimate_yaw_align_to_mask_recovers_rotation(self):
        cloud = _make_mug_like_point_cloud()
        clouds_norm, _ = normalize_clouds_to_bounds([cloud], grid_bounds=0.5, margin=0.95)
        pts = clouds_norm[0]

        # Build a cloud-frame camera at view 4.
        radius = estimate_orbit_distance_for_cloud(pts, self.intr, target_fill=0.7)
        poses = make_cloud_orbital_poses(num_views=24, elevation_deg=30.0, radius=radius)
        cam_to_world = pose_to_matrix(poses[4])

        true_yaw = 37.0
        pts_rot = rotate_clouds_yaw([pts], yaw_deg=true_yaw)[0]
        render = render_point_cloud_view(pts_rot, cam_to_world, self.intr, point_radius_px=2, blur_sigma=0.8)
        input_mask = compute_object_mask(render, threshold=5)

        est = estimate_yaw_align_to_mask(
            pts,
            input_mask=input_mask,
            base_cam_to_world=cam_to_world,
            intrinsics=self.intr,
            prior_yaw_deg=None,
            coarse_step_deg=10.0,
            fine_step_deg=2.0,
            point_radius_px=2,
            blur_sigma=0.8,
            render_threshold=10,
        )
        self.assertLessEqual(_angle_diff(est, true_yaw), 10.0)


if __name__ == "__main__":
    unittest.main()

