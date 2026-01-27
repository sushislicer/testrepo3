"""Pyrender-based virtual tabletop simulator with decoupled rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

# Monkeypatch collections.Mapping/Set for Python 3.10+ compatibility (needed by older pyrender/networkx)
import collections
import collections.abc
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Set"):
    collections.Set = collections.abc.Set
if not hasattr(collections, "MutableSet"):
    collections.MutableSet = collections.abc.MutableSet
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

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


def generate_orbital_poses(
    num_views: int,
    radius: float,
    elevation_deg: float,
    *,
    look_at: Optional[np.ndarray] = None,
) -> List[Pose]:
    poses: List[Pose] = []
    if look_at is None:
        look_at = np.zeros(3, dtype=np.float32)
    look_at = np.asarray(look_at, dtype=np.float32).reshape(3)
    elev_rad = math.radians(elevation_deg)
    for i in range(num_views):
        az = 2 * math.pi * i / num_views
        # Spherical orbit with constant distance `radius`.
        x = radius * math.cos(az) * math.cos(elev_rad)
        y = radius * math.sin(az) * math.cos(elev_rad)
        z = radius * math.sin(elev_rad)
        pos = np.array([x, y, z], dtype=np.float32)
        poses.append(Pose(pos, look_at, np.array([0.0, 0.0, 1.0], dtype=np.float32)))
    return poses


class VirtualTabletopSimulator:
    """Offscreen renderer with graceful degradation to Mock Mode."""

    def __init__(self, config: SimulatorConfig):
        self.cfg = config
        self.look_at = np.zeros(3, dtype=np.float32)
        self.poses = generate_orbital_poses(config.num_views, config.radius, config.elevation_deg, look_at=self.look_at)
        self.renderer = None
        self.scene = None
        self.mesh = None

        if _RENDERER_AVAILABLE:
            try:
                self.mesh = self._load_mesh(config.mesh_path)
                self.look_at = self._compute_look_at(self.mesh)
                if bool(getattr(config, "auto_zoom", False)):
                    self._auto_zoom_camera()
                else:
                    self.poses = generate_orbital_poses(
                        config.num_views, config.radius, config.elevation_deg, look_at=self.look_at
                    )
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

    @staticmethod
    def _compute_look_at(mesh: trimesh.Trimesh) -> np.ndarray:
        # After normalization, the object sits on z=0. Looking at its center
        # improves viewport usage and cropping robustness.
        bounds = np.asarray(mesh.bounds, dtype=np.float32)
        look_at = bounds.mean(axis=0)
        look_at[0] = 0.0
        look_at[1] = 0.0
        return look_at.astype(np.float32)

    def _auto_zoom_camera(self) -> None:
        """Set camera distance so the object occupies a large image fraction."""
        if self.mesh is None:
            return
        intr = self.cfg.intrinsics
        bounds = np.asarray(self.mesh.bounds, dtype=np.float32)
        ext = bounds[1] - bounds[0]
        # Use a conservative radius that bounds the mesh.
        obj_radius = 0.5 * float(np.max(ext))
        obj_radius = max(obj_radius, 1e-4)

        target_fill = float(np.clip(getattr(self.cfg, "target_fill", 0.65), 0.1, 0.95))
        min_dim = float(min(intr.width, intr.height))
        f = float(min(intr.fx, intr.fy))
        half_px = 0.5 * target_fill * min_dim
        # Convert desired pixel extent to an angular extent, then solve distance.
        alpha = math.atan2(half_px, f)
        distance = obj_radius / max(math.tan(alpha), 1e-6)
        # Add a small margin so the object isn't clipped.
        distance *= 1.15
        # Clamp to a reasonable range.
        distance = float(np.clip(distance, 0.15, 5.0))

        self.poses = generate_orbital_poses(
            self.cfg.num_views, distance, self.cfg.elevation_deg, look_at=self.look_at
        )

    def _build_base_scene(self, mesh: trimesh.Trimesh, cfg: SimulatorConfig) -> pyrender.Scene:
        scene = pyrender.Scene(
            bg_color=np.array(cfg.background_color + [0.0]),
            ambient_light=np.array([cfg.ambient_light] * 3, dtype=np.float32),
        )
        mesh_node = pyrender.Mesh.from_trimesh(mesh, smooth=False)
        scene.add(mesh_node)
        # Key light + fill lights to avoid near-black renders that break cropping.
        key = pyrender.DirectionalLight(color=np.ones(3), intensity=cfg.light_intensity)
        fill = pyrender.PointLight(color=np.ones(3), intensity=0.9 * cfg.light_intensity)
        rim = pyrender.PointLight(color=np.ones(3), intensity=0.7 * cfg.light_intensity)

        scene.add(key, pose=np.eye(4))
        fill_pose = np.eye(4, dtype=np.float32)
        fill_pose[:3, 3] = np.array([0.5, -0.5, 0.8], dtype=np.float32)
        scene.add(fill, pose=fill_pose)
        rim_pose = np.eye(4, dtype=np.float32)
        rim_pose[:3, 3] = np.array([-0.6, 0.6, 0.9], dtype=np.float32)
        scene.add(rim, pose=rim_pose)
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
            rgb = color[..., :3].astype(np.float32)

            if bool(getattr(self.cfg, "auto_exposure", False)) and color.shape[-1] == 4:
                alpha = color[..., 3].astype(np.float32) / 255.0
                mask = alpha > 1e-3
                if np.any(mask):
                    mean_val = float(rgb[mask].mean())
                    target = float(getattr(self.cfg, "exposure_target_mean", 80.0))
                    max_gain = float(getattr(self.cfg, "exposure_max_gain", 8.0))
                    gain = target / max(mean_val, 1e-3)
                    gain = float(np.clip(gain, 1.0, max_gain))
                    rgb[mask] = np.clip(rgb[mask] * gain, 0.0, 255.0)

            return rgb.astype(np.uint8), depth if return_depth else None
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

    def close(self) -> None:
        """Clean up renderer resources."""
        if self.renderer is not None:
            self.renderer.delete()
            self.renderer = None
