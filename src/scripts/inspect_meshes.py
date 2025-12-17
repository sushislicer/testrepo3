"""Inspect 3D Meshes with selectable backend (Open3D/Pyrender) and headless fallback."""

from __future__ import annotations

import argparse
import sys
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Path Setup
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets" / "meshes"
OUTPUT_DIR = PROJECT_ROOT / "outputs" / "mesh_inspection"

# Numpy 2.0 Compatibility Patch
if not hasattr(np, 'infty'):
    np.infty = np.inf

# Optional Imports
try:
    import open3d as o3d
except ImportError:
    o3d = None

try:
    import trimesh
    import pyrender
except ImportError:
    trimesh = None
    pyrender = None


def get_file_list(search_path: Path) -> list[Path]:
    valid_extensions = {'.obj', '.ply', '.stl', '.glb', '.gltf', '.off', '.dae'}
    if not search_path.exists():
        print(f"Error: Path not found: {search_path}")
        return []
    if search_path.is_file():
        return [search_path]
    print(f"Scanning: {search_path}")
    return sorted([p for p in search_path.rglob("*") if p.is_file() and p.suffix.lower() in valid_extensions])


def save_fallback_diagram(vertices: np.ndarray, file_name: str, backend: str):
    """Saves 2D projection diagram if window fails."""
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    save_path = OUTPUT_DIR / f"{file_name}_{backend}_fallback.png"
    
    # Downsample for speed
    pts = vertices
    if len(pts) > 10000:
        idx = np.random.choice(len(pts), 10000, replace=False)
        pts = pts[idx]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Top View (XY)
    ax1.scatter(pts[:, 0], pts[:, 1], s=0.5, alpha=0.5)
    ax1.set_title("Top View (XY)")
    ax1.axis('equal')

    # Front View (XZ)
    ax2.scatter(pts[:, 0], pts[:, 2], s=0.5, alpha=0.5)
    ax2.set_title("Front View (XZ)")
    ax2.axis('equal')

    plt.suptitle(f"Inspection ({backend}): {file_name}")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"[Headless] Saved fallback diagram to: {save_path}")


def inspect_with_open3d(file_path: Path):
    if o3d is None:
        print("Open3D not installed.")
        return

    print(f"Inspecting (Open3D): {file_path.name}")
    try:
        mesh = o3d.io.read_triangle_mesh(str(file_path))
    except Exception as e:
        print(f"Load error: {e}")
        return

    if not mesh.has_vertices():
        print("Empty mesh.")
        return

    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    extent = bbox.get_max_bound() - bbox.get_min_bound()
    center = mesh.get_center()

    print(f"{'-'*20}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Faces:    {len(mesh.triangles)}")
    print(f"Center:   {center}")
    print(f"Extent:   {extent}")

    # Visualization
    geoms = [mesh, o3d.geometry.TriangleMesh.create_coordinate_frame(size=np.max(extent)*0.5)]
    try:
        print("Attempting window...")
        o3d.visualization.draw_geometries(geoms, window_name=file_path.name, width=800, height=600)
    except Exception as e:
        print(f"Window unavailable ({e}). Saving fallback...")
        save_fallback_diagram(np.asarray(mesh.vertices), file_path.stem, "open3d")


def inspect_with_pyrender(file_path: Path):
    if trimesh is None or pyrender is None:
        print("Trimesh/Pyrender not installed.")
        return

    print(f"Inspecting (Pyrender): {file_path.name}")
    try:
        t_mesh = trimesh.load(str(file_path), force='mesh')
    except Exception as e:
        print(f"Load error: {e}")
        return

    # Handle Scenes
    if isinstance(t_mesh, trimesh.Scene):
        if not t_mesh.geometry:
            print("Empty scene.")
            return
        t_mesh = trimesh.util.concatenate(list(t_mesh.geometry.values()))

    print(f"{'-'*20}")
    print(f"Vertices: {len(t_mesh.vertices)}")
    print(f"Faces:    {len(t_mesh.faces)}")
    print(f"Center:   {t_mesh.centroid}")
    print(f"Bounds:   \n{t_mesh.bounds}")

    # Visualization
    scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1])
    try:
        mesh = pyrender.Mesh.from_trimesh(t_mesh)
        scene.add(mesh)
        
        # Add axis
        axis_len = np.max(t_mesh.extents) * 0.75
        axis_mesh = trimesh.creation.axis(axis_length=axis_len)
        pr_axis = pyrender.Mesh.from_trimesh(axis_mesh, smooth=False)
        scene.add(pr_axis)
        
        print("Attempting viewer...")
        pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(800, 600))
    except Exception as e:
        print(f"Window unavailable ({e}). Saving fallback...")
        save_fallback_diagram(t_mesh.vertices, file_path.stem, "pyrender")


def main():
    parser = argparse.ArgumentParser(description="Inspect 3D Meshes.")
    parser.add_argument("path", nargs='?', default=str(DEFAULT_ASSETS_DIR), help="Path to mesh/dir.")
    parser.add_argument("--backend", choices=['open3d', 'pyrender'], default='open3d', help="Visualization backend.")
    args = parser.parse_args()

    # Path Resolution
    target = Path(args.path)
    if not target.exists():
        target = DEFAULT_ASSETS_DIR / args.path
    
    files = get_file_list(target)
    if not files:
        print("No files found.")
        return

    print(f"Found {len(files)} files. Backend: {args.backend}")
    
    for f in files:
        if args.backend == 'open3d':
            inspect_with_open3d(f)
        else:
            inspect_with_pyrender(f)


if __name__ == "__main__":
    main()