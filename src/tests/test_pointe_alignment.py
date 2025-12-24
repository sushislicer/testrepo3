"""
Unit tests for robust multi-seed Point-E alignment.
Usage: python -m src.tests.test_pointe_alignment
"""

import math
import unittest

import numpy as np
from scipy.spatial import cKDTree

from src.pointe_alignment import AlignmentParams, align_clouds_to_reference_front


def _rot_z(deg: float) -> np.ndarray:
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def _angle_xy(point: np.ndarray) -> float:
    # Angle with 0 at +Y (front), increasing towards +X.
    return math.atan2(float(point[0]), float(point[1]))


def _wrap_angle(rad: float) -> float:
    return (rad + math.pi) % (2 * math.pi) - math.pi


def _make_mug_cloud(handle_yaw_deg: float, *, rng: np.random.RandomState) -> np.ndarray:
    # Body: upright cylinder around origin.
    n_body = 5000
    theta = rng.uniform(0.0, 2 * math.pi, size=n_body).astype(np.float32)
    z = rng.uniform(-0.5, 0.5, size=n_body).astype(np.float32)
    r = 1.0 + 0.02 * rng.randn(n_body).astype(np.float32)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    body = np.stack([x, y, z], axis=1).astype(np.float32)

    # "Logo" patch on the front face (+Y). This is a small asymmetry that should
    # anchor yaw alignment (mimics visible texture/structure in the conditioning view).
    n_logo = 400
    logo_x = 0.15 * rng.randn(n_logo).astype(np.float32)
    logo_y = np.full(n_logo, 1.02, dtype=np.float32) + 0.02 * rng.randn(n_logo).astype(np.float32)
    logo_z = 0.15 * rng.randn(n_logo).astype(np.float32)
    logo = np.stack([logo_x, logo_y, logo_z], axis=1).astype(np.float32)

    # Handle: a small circle offset from the body at some yaw around Z.
    n_handle = 600
    phi = rng.uniform(0.0, 2 * math.pi, size=n_handle).astype(np.float32)
    handle_r = 0.18 + 0.01 * rng.randn(n_handle).astype(np.float32)
    yaw = math.radians(handle_yaw_deg)
    hx0 = 1.25 * math.sin(yaw)  # note: 0deg is +Y, so x uses sin, y uses cos
    hy0 = 1.25 * math.cos(yaw)
    hx = hx0 + handle_r * np.cos(phi)
    hy = hy0 + handle_r * np.sin(phi)
    hz = 0.25 * np.sin(phi)  # some vertical extent
    handle = np.stack([hx, hy, hz], axis=1).astype(np.float32)

    return np.concatenate([body, logo, handle], axis=0)


class TestPointEAlignment(unittest.TestCase):
    def test_align_front_face_across_seeds(self):
        rng = np.random.RandomState(0)

        # Reference: handle is "behind" (180deg). Other seeds hallucinate handles at
        # different back-half angles, and have small global pose jitter.
        handle_yaws = [180.0, 150.0, 210.0, 240.0, 120.0]
        clouds = []
        for i, hy in enumerate(handle_yaws):
            cloud = _make_mug_cloud(hy, rng=rng)

            # Apply seed-specific small jitter: translation, scale, yaw.
            yaw_jitter = float(rng.uniform(-8.0, 8.0))
            scale = float(rng.uniform(0.92, 1.08))
            trans = rng.uniform(-0.12, 0.12, size=3).astype(np.float32)
            trans[2] *= 0.3  # smaller Z drift

            R = _rot_z(yaw_jitter)
            cloud = (cloud @ R.T) * scale + trans[None, :]
            clouds.append(cloud.astype(np.float32))

        params = AlignmentParams(
            max_yaw_deg=30.0,
            yaw_step_deg=5.0,
            trim_ratio=0.7,
            core_ratio=1.0,
            surface_grid=96,
            max_points=2500,
        )
        aligned = align_clouds_to_reference_front(clouds, params=params, reference_idx=0, rng_seed=0)
        self.assertEqual(len(aligned), len(clouds))

        n_body = 5000
        n_logo = 400
        logo_slice = slice(n_body, n_body + n_logo)
        handle_slice = slice(n_body + n_logo, None)

        # 1) Visible "logo" front patch should align close to reference (angle near 0).
        ref_logo_ctr = aligned[0][logo_slice].mean(axis=0)
        ref_logo_ang = _angle_xy(ref_logo_ctr)
        for i in range(1, len(aligned)):
            logo_ctr = aligned[i][logo_slice].mean(axis=0)
            ang = _angle_xy(logo_ctr)
            self.assertLess(abs(_wrap_angle(ang - ref_logo_ang)), math.radians(20.0))

        # 2) Handle angles should still vary across seeds (alignment shouldn't collapse them).
        handle_angles = []
        for i in range(len(aligned)):
            hctr = aligned[i][handle_slice].mean(axis=0)
            handle_angles.append(_angle_xy(hctr))
        # Use circular spread as a simple check: max pairwise diff should be large.
        diffs = [abs(_wrap_angle(a - b)) for a in handle_angles for b in handle_angles]
        self.assertGreater(max(diffs), math.radians(35.0))

        # 3) Body points should tightly match the reference after alignment.
        ref_body = aligned[0][:n_body]
        tree = cKDTree(ref_body)
        for i in range(1, len(aligned)):
            body = aligned[i][:n_body]
            d, _ = tree.query(body, k=1)
            self.assertLess(float(np.mean(d)), 0.08)


if __name__ == "__main__":
    unittest.main()
