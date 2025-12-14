from __future__ import annotations

import trimesh
import pyrender
import numpy as np
import argparse
from pathlib import Path

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
CURRENT_SCRIPT_DIR = CURRENT_SCRIPT_PATH.parent
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent.parent
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets" / "meshes"

# Disturbing monkey patch cause compatibility
if not hasattr(np, 'infty'):
    np.infty = np.inf

def get_file_list(search_path: Path) -> list[Path]:
    valid_extensions = {'.obj', '.ply', '.stl', '.glb', '.gltf', '.off', '.dae'}
    
    if not search_path.exists():
        print(f"Error: Path not found: {search_path}")
        return []

    if search_path.is_file():
        return [search_path]
    
    print(f"Scanning directory: {search_path}")
    
    # Recursive search
    mesh_files = [
        p for p in search_path.rglob("*") 
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
            
    return sorted(mesh_files)

def analyze_and_visualize(file_path: Path):
    print(f"\n{'='*40}")
    print(f"Inspecting: {file_path.name}")
    
    try:
        t_mesh = trimesh.load(str(file_path))
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if isinstance(t_mesh, trimesh.Scene):
        print("  -> Loaded as Scene, concatenating geometry for inspection...")
        geometries = list(t_mesh.geometry.values())
        if not geometries:
            print("Error: Scene is empty.")
            return
        t_mesh = trimesh.util.concatenate(geometries)

    vertices = t_mesh.vertices
    faces = t_mesh.faces
    bounds = t_mesh.bounds
    extents = t_mesh.extents
    center = t_mesh.centroid
    max_dim = np.max(extents)

    print(f"{'-'*20}")
    print(f"Vertices: {len(vertices)}")
    print(f"Faces:    {len(faces)}")
    print(f"Center:   {center}")
    print(f"Extents:  {extents}")
    print(f"Bounds:   \n{bounds}")

    scene = pyrender.Scene(bg_color=[0.1, 0.1, 0.1])
    try:
        mesh = pyrender.Mesh.from_trimesh(t_mesh)
        scene.add(mesh)
    except Exception as e:
        print(f"Error converting to pyrender mesh: {e}")
        return

    axis_len = max(max_dim * 0.75, 0.1)
    axis_mesh = trimesh.creation.axis(origin_size=axis_len*0.05, axis_length=axis_len)
    
    try:
        if isinstance(axis_mesh, trimesh.Scene):
             axis_mesh = trimesh.util.concatenate(list(axis_mesh.geometry.values()))
        
        pr_axis = pyrender.Mesh.from_trimesh(axis_mesh, smooth=False)
        scene.add(pr_axis)
    except Exception as e:
        print(f"Could not add axis frame: {e}")

    print("\nStarting Viewer...")
    print("Controls:")
    print("  [Mouse Left]   Rotate")
    print("  [Mouse Right]  Pan")
    print("  [Scroll]       Zoom")
    print("  [A]            Toggle Axis (if supported by viewer)")
    print("  [I]            Toggle Inspector")
    print("  [Close Window] Next File")

    pyrender.Viewer(scene, use_raymond_lighting=True, viewport_size=(1024, 768), caption=f"Inspecting: {file_path.name}")

def main():
    parser = argparse.ArgumentParser(description="Inspect 3D Meshes (Pyrender version).")
    parser.add_argument(
        "path", 
        nargs='?',
        default=str(DEFAULT_ASSETS_DIR),
        help=f"Path to mesh file or directory. Defaults to: {DEFAULT_ASSETS_DIR}"
    )
    
    args = parser.parse_args()
    target_path = Path(args.path)

    print(f"Target path: {target_path}")
    files = get_file_list(target_path)
    
    if not files:
        print("\n No mesh files found!")
        if target_path == DEFAULT_ASSETS_DIR:
            print(f"Ensure your meshes are in: {DEFAULT_ASSETS_DIR}")
        return
    
    print(f"Found {len(files)} files.")
    for f in files:
        analyze_and_visualize(f)

if __name__ == "__main__":
    main()