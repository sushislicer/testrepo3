from __future__ import annotations

import open3d as o3d
import numpy as np
import argparse
from pathlib import Path

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
CURRENT_SCRIPT_DIR = CURRENT_SCRIPT_PATH.parent  # root/src/scripts
PROJECT_ROOT = CURRENT_SCRIPT_DIR.parent.parent
DEFAULT_ASSETS_DIR = PROJECT_ROOT / "assets" / "meshes"

def get_file_list(search_path: Path) -> list[Path]:
    
    # Search function for meshes from google
    valid_extensions = {'.obj', '.ply', '.stl', '.glb', '.gltf', '.off'}
    if not search_path.exists():
        print(f"Error: Path not found: {search_path}")
        return []

    if search_path.is_file():
        return [search_path]
    
    print(f"Scanning directory: {search_path}")
    
    mesh_files = [
        p for p in search_path.rglob("*") 
        if p.is_file() and p.suffix.lower() in valid_extensions
    ]
            
    return sorted(mesh_files)

def analyze_and_visualize(file_path: Path):
    print(f"Inspecting: {file_path.name}")
    try:
        mesh = o3d.io.read_triangle_mesh(str(file_path))
    except Exception as e:
        print(f"Error loading file: {e}")
        return

    if not mesh.has_vertices():
        print("Error: Empty mesh (no vertices) or file format not supported.")
        print("Try installing: pip install open3d --upgrade")
        return

    mesh.compute_vertex_normals()
    bbox = mesh.get_axis_aligned_bounding_box()
    min_bound = bbox.get_min_bound()
    max_bound = bbox.get_max_bound()
    extent = max_bound - min_bound
    center = mesh.get_center()
    max_dim = np.max(extent)

    # DEBUG
    print(f"{'-'*20}")
    print(f"Vertices: {len(mesh.vertices)}")
    print(f"Triangles: {len(mesh.triangles)}")
    print(f"Center (XYZ): [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"Dimensions (XYZ): [{extent[0]:.4f}, {extent[1]:.4f}, {extent[2]:.4f}]")
    
    # Coordinates at origin
    geometries = [mesh]
    frame_size = max(max_dim * 0.5, 0.1) 
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=frame_size, origin=[0, 0, 0])
    geometries.append(coord_frame)

    # Visual Controls
    print("\nControls:")
    print("  [Mouse] Rotate/Pan/Zoom")
    print("  [Q] or [Esc] Close window and go to next file")
    print("  [W] Toggle Wireframe")
    print("  [+/-] Increase/Decrease Point Size")
    
    o3d.visualization.draw_geometries(
        geometries,
        window_name=f"Inspecting: {file_path.name}",
        width=1024,
        height=768,
        left=50,
        top=50,
        mesh_show_back_face=True
    )

def main():
    parser = argparse.ArgumentParser(description="Inspect 3D Meshes for CV/Robotics.")
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