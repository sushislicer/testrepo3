"""Pyrender-based virtual tabletop simulator with decoupled rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Decoupled Import: Soft-fail if renderer/system libs are missing
try:
    import trimesh
    import pyrender
    _RENDERER_AVAILABLE = True
except (ImportError, OSError) as e:
    print(f"[Simulator] Rendering libraries missing ({e}). Running in Mock Mode.")
    _RENDERER_AVAILABLE = False

from .config import SimulatorConfig, CameraIntrinsics


@dataclass
class Pose:
    position: np.ndarray
    look_at: np.ndarray
    up: np.ndarray


def pose_to_matrix(pose: Pose) -> np.ndarray:
    forward = pose.look_at - pose.position
    forward = forward / (np.linalg.norm(forward) + 1e-8)
    right = np.cross(forward, pose.up)
    right = right / (np.linalg.norm(right) + 1e-8)
    up = np.cross(right, forward)
    cam_to_world = np.eye(4, dtype=np.float32)
    cam_to_world[:3, 0] = right
    cam_to_world[:3, 1] = up
    cam_to_world[:3, 2] = -forward
    cam_to_world[:3, 3] = pose.position
    return cam_to_world


def generate_orbital_poses(num_views: int, radius: float, elevation_deg: float) -> List[Pose]:
    poses: List[Pose] = []
    elev_rad = math.radians(elevation_deg)
    for i in range(num_views):
        az = 2 * math.pi * i / num_views
        x = radius * math.cos(az)
        y = radius * math.sin(az)
        z = radius * math.sin(elev_rad)
        pos = np.array([x, y, z], dtype=np.float32)
        poses.append(Pose(pos, np.zeros(3, dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32)))
    return poses


class VirtualTabletopSimulator:
    """Offscreen renderer with graceful degradation to Mock Mode."""

    def __init__(self, config: SimulatorConfig):
        self.cfg = config
        self.poses = generate_orbital_poses(config.num_views, config.radius, config.elevation_deg)
        self.renderer = None
        self.scene = None
        self.mesh = None

        if _RENDERER_AVAILABLE:
            try:
                self.mesh = self._load_mesh(config.mesh_path)
                self.scene = self._build_base_scene(self.mesh, config)
                self.renderer = pyrender.OffscreenRenderer(
                    viewport_width=config.intrinsics.width, 
                    viewport_height=config.intrinsics.height
                )
            except Exception as e:
                print(f"[Simulator] Renderer init failed ({e}). Mock Mode enabled.")
                self.renderer = None

    def _load_mesh(self, mesh_path: str) -> trimesh.Trimesh:
        path = Path(mesh_path)
        if not path.exists():
            raise FileNotFoundError(f"Mesh not found: {mesh_path}")
        mesh = trimesh.load_mesh(str(path), force='mesh')
        if isinstance(mesh, trimesh.Scene):
            mesh = next(iter(mesh.geometry.values()))
        
        # Normalize to table
        bounds = mesh.bounds
        min_xyz, max_xyz = bounds
        translate = np.array([-0.5*(min_xyz[0]+max_xyz[0]), -0.5*(min_xyz[1]+max_xyz[1]), -min_xyz[2]])
        mesh.apply_translation(translate)
        return mesh

    def _build_base_scene(self, mesh: trimesh.Trimesh, cfg: SimulatorConfig) -> pyrender.Scene:
        scene = pyrender.Scene(bg_color=np.array(cfg.background_color + [0.0]))
        mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(mesh_node)
        light = pyrender.DirectionalLight(color=np.ones(3), intensity=cfg.light_intensity)
        scene.add(light, pose=np.eye(4))
        return scene

    def render_view(self, pose_idx: int, return_depth: bool = False) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Render RGB. Returns dummy image if renderer unavailable."""
        if self.renderer:
            pose = self.poses[pose_idx % len(self.poses)]
            intr = self.cfg.intrinsics
            cam = pyrender.IntrinsicsCamera(fx=intr.fx, fy=intr.fy, cx=intr.cx, cy=intr.cy, znear=0.05, zfar=5.0)
            cam_node = self.scene.add(cam, pose=pose_to_matrix(pose))
            
            color, depth = self.renderer.render(self.scene, flags=pyrender.RenderFlags.RGBA)
            self.scene.remove_node(cam_node)
            return color[..., :3], depth if return_depth else None
        else:
            # Mock Mode: Return a gray box so pipelines don't crash
            h, w = self.cfg.intrinsics.height, self.cfg.intrinsics.width
            mock = np.zeros((h, w, 3), dtype=np.uint8)
            mock[h//3:2*h//3, w//3:2*w//3] = 100
            return mock, None

    def list_poses(self) -> List[Pose]:
        return self.poses

    def get_pose_matrix(self, idx: int) -> np.ndarray:
        return pose_to_matrix(self.poses[idx % len(self.poses)])