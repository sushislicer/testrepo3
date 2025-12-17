"""
Load a directory of meshes, render them together, and save to assets/images.
Preserves relative positions between meshes.
"""

from __future__ import annotations

import argparse
import sys
import numpy as np
import trimesh
import pyrender
import colorsys
from pathlib import Path
from PIL import Image

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

MESHES_ROOT = PROJECT_ROOT / "assets" / "meshes"
IMAGES_DIR = PROJECT_ROOT / "assets" / "images"
VALID_EXTS = {'.obj', '.ply', '.stl', '.glb', '.gltf', '.off'}


def load_mesh_set(directory: Path) -> list[trimesh.Trimesh]:
    """Loads all valid meshes in the directory."""
    files = [p for p in directory.glob("*") if p.suffix.lower() in VALID_EXTS]
    meshes = []
    
    if not files:
        raise FileNotFoundError(f"No mesh files found in {directory}")

    print(f"Found {len(files)} parts in {directory.name}:")
    for f in sorted(files):
        print(f"  - {f.name}")
        m = trimesh.load(str(f), force='mesh')
        # Handle scenes by concatenating
        if isinstance(m, trimesh.Scene):
            if not m.geometry: continue
            m = trimesh.util.concatenate(tuple(m.geometry.values()))
        meshes.append(m)
    return meshes


def normalize_group(meshes: list[trimesh.Trimesh]) -> list[trimesh.Trimesh]:
    """Centers and scales the entire group while keeping relative positions."""
    if not meshes: return []

    # 1. Combine temporarily to find total bounds
    combined = trimesh.util.concatenate(meshes)
    centroid = combined.centroid
    max_extent = np.max(combined.extents)
    
    # 2. Apply transform to each individual mesh
    scale = 1.5 / max_extent if max_extent > 0 else 1.0
    
    for m in meshes:
        m.apply_translation(-centroid)
        m.apply_scale(scale)
        
    return meshes


def render_scene(meshes: list[trimesh.Trimesh], output_name: str) -> None:
    scene = pyrender.Scene(bg_color=[1.0, 1.0, 1.0])

    # Add meshes with distinct colors
    for i, m in enumerate(meshes):
        # Generate color based on index
        hue = i / max(len(meshes), 1)
        rgb = colorsys.hsv_to_rgb(hue, 0.6, 0.9)
        mat = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=[rgb[0], rgb[1], rgb[2], 1.0],
            metallicFactor=0.1,
            roughnessFactor=0.6
        )
        pr_mesh = pyrender.Mesh.from_trimesh(m, material=mat)
        scene.add(pr_mesh)

    # Lighting
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=np.eye(4))

    # Camera
    camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
    cam_pose = np.eye(4)
    cam_pose[:3, 3] = [0.8, 0.8, 2.0] # Offset slightly to see depth
    
    # Look at origin logic
    target = np.array([0, 0, 0])
    position = cam_pose[:3, 3]
    forward = target - position
    forward /= np.linalg.norm(forward)
    up = np.array([0, 1, 0])
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    
    cam_pose[:3, 0] = right
    cam_pose[:3, 1] = new_up
    cam_pose[:3, 2] = -forward
    scene.add(camera, pose=cam_pose)

    # Render
    r = pyrender.OffscreenRenderer(512, 512)
    rgb, _ = r.render(scene)
    r.delete()

    # Save
    IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = IMAGES_DIR / f"{output_name}.png"
    Image.fromarray(rgb).save(out_path)
    print(f"Saved combined render to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Render a folder of meshes.")
    parser.add_argument("folder", type=str, help="Subfolder in assets/meshes (e.g., 'object1')")
    parser.add_argument("--name", type=str, default=None, help="Output filename.")
    args = parser.parse_args()

    target_dir = Path(args.folder)
    # Check absolute or relative to assets/meshes
    if not target_dir.exists():
        target_dir = MESHES_ROOT / args.folder
    
    if not target_dir.exists() or not target_dir.is_dir():
        print(f"Directory not found: {target_dir}")
        return

    out_name = args.name if args.name else target_dir.name

    try:
        meshes = load_mesh_set(target_dir)
        meshes = normalize_group(meshes)
        render_scene(meshes, out_name)
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()