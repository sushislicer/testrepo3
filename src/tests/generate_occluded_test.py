"""
Generates a partially occluded mug mesh.
Fallback: Uses Matplotlib to render a 'sketch' if Pyrender fails.
"""

from __future__ import annotations

import os
# Force EGL for Colab before imports
os.environ["PYOPENGL_PLATFORM"] = "egl"

import sys
import numpy as np
import trimesh
import imageio
import matplotlib.pyplot as plt
from pathlib import Path

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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


def render_fallback_matplotlib(mesh: trimesh.Trimesh, out_path: Path):
    """Renders a simple point cloud sketch if Pyrender fails."""
    print("  [Fallback] Rendering mesh sketch via Matplotlib...")
    
    points = mesh.sample(5000)
    x = points[:, 0]
    y = points[:, 1] # Front view projection
    
    plt.figure(figsize=(5, 5))
    plt.scatter(x, y, c='black', s=1)
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout(pad=0)
    
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=100)
    plt.close()
    print(f"  [Fallback] Saved sketch to {out_path}")


def main():
    MESH_OUT.parent.mkdir(parents=True, exist_ok=True)
    IMAGE_OUT.parent.mkdir(parents=True, exist_ok=True)

    print("Generating mesh...")
    mesh = create_partial_mug()
    mesh.export(MESH_OUT)
    print(f"Saved mesh: {MESH_OUT}")

    print("Attempting to render view...")
    render_success = False
    
    try:
        from src.config import ActiveHallucinationConfig
        from src.simulator import VirtualTabletopSimulator

        cfg = ActiveHallucinationConfig()
        cfg.simulator.mesh_path = str(MESH_OUT)
        cfg.simulator.intrinsics.width = 512
        cfg.simulator.intrinsics.height = 512
        cfg.simulator.background_color = [1.0, 1.0, 1.0]

        sim = VirtualTabletopSimulator(cfg.simulator)
        rgb, _ = sim.render_view(idx=0)
        imageio.imwrite(IMAGE_OUT, rgb.astype(np.uint8))
        print(f"Saved HQ image: {IMAGE_OUT}")
        render_success = True

    except Exception as e:
        print(f"[Warning] Pyrender failed ({e}). Using fallback.")
        
    if not render_success:
        render_fallback_matplotlib(mesh, IMAGE_OUT)


if __name__ == "__main__":
    main()