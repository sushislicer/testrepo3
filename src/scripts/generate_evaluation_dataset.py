"""Generates a diverse dataset (Mugs, Pans, Hammers) for evaluation."""

from __future__ import annotations

import sys
import numpy as np
import trimesh
from pathlib import Path

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

ASSETS_DIR = PROJECT_ROOT / "assets" / "meshes"


def normalize_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    mesh.apply_translation(-mesh.centroid)
    mesh.apply_scale(1.0 / np.max(mesh.extents))
    bounds = mesh.bounds
    mesh.apply_translation([0, 0, -bounds[0][2]])
    return mesh


def create_mug(rng: np.random.RandomState) -> trimesh.Trimesh:
    radius = rng.uniform(0.3, 0.45)
    height = rng.uniform(0.5, 0.8)
    body = trimesh.creation.cylinder(radius=radius, height=height)
    
    if rng.rand() > 0.5:
        # Curved handle
        h_rad, h_out = rng.uniform(0.05, 0.08), rng.uniform(0.15, 0.25)
        handle = trimesh.creation.annulus(r_min=h_rad, r_max=h_out, height=rng.uniform(0.3, 0.5))
        handle.apply_transform(trimesh.transformations.rotation_matrix(np.pi/2, [1, 0, 0]))
        handle.apply_translation([radius + h_rad, 0, 0])
    else:
        # Square handle
        ext = [rng.uniform(0.2, 0.3), 0.08, height * 0.6]
        handle = trimesh.creation.box(extents=ext)
        handle.apply_translation([radius + ext[0]/2 - 0.02, 0, 0])

    mesh = trimesh.util.concatenate([body, handle])
    mesh.visual.face_colors = [rng.randint(50, 200) for _ in range(3)] + [255]
    return normalize_mesh(mesh)


def create_pan(rng: np.random.RandomState) -> trimesh.Trimesh:
    radius = rng.uniform(0.35, 0.5)
    height = rng.uniform(0.08, 0.15)
    base = trimesh.creation.cylinder(radius=radius, height=height)
    
    handle_len = rng.uniform(0.7, 1.2)
    handle_thick = rng.uniform(0.06, 0.1)
    handle = trimesh.creation.box(extents=[handle_len, handle_thick, handle_thick])
    handle.apply_translation([radius + handle_len/2, 0, height/2])
    
    mesh = trimesh.util.concatenate([base, handle])
    mesh.visual.face_colors = [50, 50, 50, 255]
    return normalize_mesh(mesh)


def create_hammer(rng: np.random.RandomState) -> trimesh.Trimesh:
    head_x = rng.uniform(0.2, 0.35)
    head_yz = rng.uniform(0.1, 0.15)
    head = trimesh.creation.box(extents=[head_x, head_yz, head_yz])
    
    handle_len = rng.uniform(0.6, 0.9)
    handle = trimesh.creation.cylinder(radius=0.04, height=handle_len)
    
    # Position head on top of handle
    head.apply_translation([0, 0, handle_len/2 + head_yz/2])
    
    mesh = trimesh.util.concatenate([head, handle])
    mesh.visual.face_colors = [100, 100, 150, 255]
    return normalize_mesh(mesh)


def main():
    ASSETS_DIR.mkdir(parents=True, exist_ok=True)
    categories = {
        "mugs": create_mug,
        "pans": create_pan,
        "tools": create_hammer
    }

    print(f"Generating evaluation dataset in {ASSETS_DIR}...")

    for cat, creator_func in categories.items():
        cat_dir = ASSETS_DIR / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Generating {cat}...")
        
        for i in range(10):
            rng = np.random.RandomState(i)
            mesh = creator_func(rng)
            filename = cat_dir / f"{cat[:-1]}_{i:02d}.obj"
            mesh.export(filename)
    
    print("\n[Ready] Run batch experiments:")
    print('python src/scripts/run_experiments.py --mesh_glob "assets/meshes/**/*.obj" --trials 2')


if __name__ == "__main__":
    main()