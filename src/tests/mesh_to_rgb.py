"""Convert a mesh from assets/meshes to an RGB image in assets/images."""

from __future__ import annotations

import argparse
import sys
import numpy as np
import trimesh
import pyrender
from pathlib import Path
from PIL import Image

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MESHES_DIR = PROJECT_ROOT / "assets" / "meshes"
IMAGES_DIR = PROJECT_ROOT / "assets" / "images"


def get_rgb_image_object(mesh_filename: str, resolution: int = 512) -> np.ndarray:
    """Loads mesh, normalizes, renders, returns RGB array."""
    mesh_path = MESHES_DIR / mesh_filename
    if not mesh_path.exists():
        # Try fuzzy match
        candidates = list(MESHES_DIR.glob(f"{mesh_filename}*.obj"))
        if candidates:
            mesh_path = candidates[0]
        else:
            raise FileNotFoundError(f"Mesh not found in {MESHES_DIR}: {mesh_filename}")

    print(f"Loading: {mesh_path.name}")
    mesh = trimesh.load(str(mesh_path), force='mesh')
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.geometry.values()))

    # Normalize to unit sphere at origin
    mesh.apply_translation(-mesh.centroid)
    max_extent = np.max(mesh.extents)
    if max_extent > 0:
        mesh.apply_scale(1.5 / max_extent)

    # Setup Scene
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])
    
    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=[0.7, 0.7, 0.7, 1.0],
        metallicFactor=0.0,
        roughnessFactor=0.5
    )
    pr_mesh = pyrender.Mesh.from_trimesh(mesh, material=material)
    scene.add(pr_mesh)

    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0, 0, 2.0]
    scene.add(camera, pose=cam_pose)

    # Render
    r = pyrender.OffscreenRenderer(resolution, resolution)
    rgb, _ = r.render(scene)
    r.delete()

    return rgb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mesh", type=str, default=None, help="Name of .obj in assets/meshes")
    args = parser.parse_args()

    # Default to first available mesh if none specified
    target_mesh = args.mesh
    if not target_mesh:
        objs = list(MESHES_DIR.glob("*.obj"))
        if not objs:
            print(f"No .obj files found in {MESHES_DIR}")
            return
        target_mesh = objs[0].name

    try:
        rgb_array = get_rgb_image_object(target_mesh)
        
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        out_path = IMAGES_DIR / f"{Path(target_mesh).stem}_rendered.png"
        
        Image.fromarray(rgb_array).save(out_path)
        print(f"Success. Saved RGB Image to: {out_path}")

    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()