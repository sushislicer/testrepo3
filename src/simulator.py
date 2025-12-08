"""Pyrender-based virtual tabletop simulator and orbital camera utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import numpy as np

try:
    import trimesh
    import pyrender
except ImportError as exc:  # pragma: no cover - dependency warning path
    raise ImportError(
        "trimesh and pyrender are required for simulator. Install via requirements.txt"
    ) from exc

from .config import SimulatorConfig, CameraIntrinsics


@dataclass
class Pose:
    position: np.ndarray  # shape (3,)
    look_at: np.ndarray  # shape (3,)
    up: np.ndarray  # shape (3,)


def pose_to_matrix(pose: Pose) -> np.ndarray:
    """Convert Pose to a 4x4 camera-to-world transform matrix."""
    forward = pose.look_at - pose.position
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, pose.up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    cam_to_world = np.eye(4, dtype=np.float32)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = -forward  # pyrender uses -Z forward
    cam_to_world[:3, 3] = pose.position
    return cam_to_world


def generate_orbital_poses(num_views: int, radius: float, elevation_deg: float) -> List[Pose]:
    """Generate evenly spaced poses on an azimuth ring at fixed elevation."""
    poses: List[Pose] = []
    elev_rad = math.radians(elevation_deg)
    for i in range(num_views):
        az = 2 * math.pi * i / num_views
        x = radius * math.cos(az)
        y = radius * math.sin(az)
        z = radius * math.sin(elev_rad)
        pos = np.array([x, y, z], dtype=np.float32)
        poses.append(Pose(position=pos, look_at=np.zeros(3, dtype=np.float32), up=np.array([0.0, 0.0, 1.0], dtype=np.float32)))
    return poses


def _normalize_mesh_to_table(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Translate mesh so its minimum z sits on the table plane (z=0) and center xy."""
    mesh = mesh.copy()
    bounds = mesh.bounds
    min_xyz, max_xyz = bounds
    translate = np.array([-0.5 * (min_xyz[0] + max_xyz[0]), -0.5 * (min_xyz[1] + max_xyz[1]), -min_xyz[2]])
    mesh.apply_translation(translate)
    return mesh


class VirtualTabletopSimulator:
    """Offscreen renderer for tabletop objects with an orbital camera ring."""

    def __init__(self, config: SimulatorConfig):
        self.cfg = config
        self.poses = generate_orbital_poses(config.num_views, config.radius, config.elevation_deg)
        self.mesh = self._load_mesh(config.mesh_path)
        self.scene = self._build_base_scene(self.mesh, config)
        self.renderer = pyrender.OffscreenRenderer(viewport_width=config.intrinsics.width, viewport_height=config.intrinsics.height)

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        path = Path(mesh_path)
        if not path.exists():
            raise FileNotFoundError(f"Mesh path not found: {mesh_path}")
        mesh = trimesh.load_mesh(mesh_path, force='mesh')
        if not isinstance(mesh, trimesh.Trimesh) and hasattr(mesh, "geometry"):
            # Some formats load as Scene; convert first geometry.
            mesh = next(iter(mesh.geometry.values()))
        mesh = _normalize_mesh_to_table(mesh)
        return mesh

    def _build_base_scene(self, mesh: trimesh.Trimesh, cfg: SimulatorConfig) -> pyrender.Scene:
        scene = pyrender.Scene(bg_color=np.array(cfg.background_color + [0.0]), ambient_light=np.array([0.1, 0.1, 0.1, 1.0]))
        mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(mesh_node)

        light = pyrender.DirectionalLight(color=np.ones(3), intensity=cfg.light_intensity)
        scene.add(light, pose=np.eye(4))
        return scene

    def _add_camera(self, pose: Pose, intr: CameraIntrinsics) -> Tuple[pyrender.Node, pyrender.Node]:
        cam = pyrender.IntrinsicsCamera(fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy, znear=0.05, zfar=5.0)
        cam_pose = pose_to_matrix(pose)
        cam_node = self.scene.add(cam, pose=cam_pose)
        return cam, cam_node

    def render_view(self, pose_idx: int, return_depth: bool = False) -> Tuple[np.ndarray, np.ndarray | None]:
        """Render RGB (and optionally depth) from the given pose index."""
        pose = self.poses[pose_idx % len(self.poses)]
        intr = self.cfg.intrinsics
        cam, cam_node = self._add_camera(pose, intr)
        flags = pyrender.RenderFlags.RGBA
        color, depth = self.renderer.render(self.scene, flags=flags)
        self.scene.remove_node(cam_node)
        rgb = color[..., :3]
        return rgb, depth if return_depth else None

    def list_poses(self) -> List[Pose]:
        return self.poses
