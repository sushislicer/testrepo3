import open3d as o3d
import numpy as np
import argparse
import os
import sys

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_SCRIPT_DIR)
DEFAULT_ASSETS_DIR = os.path.join(PROJECT_ROOT, "assets", "meshes")

def get_file_list(path):
    valid_extensions = ['.obj', '.ply', '.stl', '.glb', '.gltf', '.off']
    mesh_files = []

    if not os.path.exists(path):
        print(f"Error: Path not found: {path}")
        return []

    if os.path.isfile(path):
        return [path]
    
    print(f"Scanning directory: {path}")
    for root, dirs, files in os.walk(path):
        for file in files:
            if any(file.lower().endswith(ext) for ext in valid_extensions):
                mesh_files.append(os.path.join(root, file))
    return sorted(mesh_files)

def analyze_and_visualize(file_path):
    print(f"Inspecting: {os.path.basename(file_path)}")
    try:
        mesh = o3d.io.read_triangle_mesh(file_path)
    except Exception as e:
        print(f"Error loading file: {e}")
        return
    if not mesh.has_vertices():
        print("Error: Empty mesh (no vertices). or file format not supported by standard loader.")
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
        window_name=f"Inspecting: {os.path.basename(file_path)}",
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
        default=DEFAULT_ASSETS_DIR, 
        help=f"Path to mesh file or directory. Defaults to: {DEFAULT_ASSETS_DIR}"
    )
    
    args = parser.parse_args()

    print(f"Target path: {args.path}")
    files = get_file_list(args.path)
    
    if not files:
        print("\n No mesh files found!")
        return
    
    for f in files:
        analyze_and_visualize(f)

if __name__ == "__main__":
    main()