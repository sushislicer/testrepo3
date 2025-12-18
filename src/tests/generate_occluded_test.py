"""Generates a partially occluded mug mesh for testing."""

from __future__ import annotations

import sys
import numpy as np
import trimesh
import imageio
from pathlib import Path

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import ActiveHallucinationConfig
from src.simulator import VirtualTabletopSimulator

ASSETS_DIR = PROJECT_ROOT / "assets"
MESH_OUT = ASSETS_DIR / "meshes" / "example.obj"
IMAGE_OUT = ASSETS_DIR / "images" / "example.png"


def create_partial_mug() -> trimesh.Trimesh:
    radius, height = 0.35, 0.7
    body = trimesh.creation.cylinder(radius=radius, height=height)

    handle = trimesh.creation.annulus(r_min=0.08, r_max=0.15, height=0.4)
    handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
    handle.apply_translation([radius + 0.1, 0, 0])

    mesh = trimesh.util.concatenate([body, handle])
    mesh.visual.face_colors = [150, 150, 150, 255]

    # Rotate 150 degrees: Handle peeks out slightly
    rot = trimesh.transformations.rotation_matrix(np.radians(150), [0, 0, 1])
    mesh.apply_transform(rot)

    return mesh


def main():
    MESH_OUT.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Generating mesh...")
    mesh = create_partial_mug()
    mesh.export(MESH_OUT)
    print(f"Saved: {MESH_OUT}")

    print("Rendering view...")
    cfg = ActiveHallucinationConfig()
    cfg.simulator.mesh_path = str(MESH_OUT)
    cfg.simulator.intrinsics.width = 512
    cfg.simulator.intrinsics.height = 512
    cfg.simulator.background_color = [1.0, 1.0, 1.0]

    try:
        sim = VirtualTabletopSimulator(cfg.simulator)
        rgb, _ = sim.render_view(idx=0)
        imageio.imwrite(IMAGE_OUT, rgb.astype(np.uint8))
        print(f"Saved: {IMAGE_OUT}")
    except Exception as e:
        print(f"Render failed: {e}")


if __name__ == "__main__":
    main()