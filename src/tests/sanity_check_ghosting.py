"""
Sanity check: Ghosting artifacts via variance field.
Location: src/tests/sanity_check_ghosting.py
Usage: python src/tests/sanity_check_ghosting.py --image assets/images/chair.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# Path Setup
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# Headless Import Safety
try:
    import open3d as o3d
except ImportError:
    o3d = None

from src.config import ActiveHallucinationConfig
from src.pointe_wrapper import PointEGenerator
# Import the specific functions and class from your implementation
from src.variance_field import compute_variance_field, export_score_points, VoxelGrid


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of Point-E seeds.")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution.")
    return parser.parse_args()


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists(): return path
    asset_path = PROJECT_ROOT / "assets" / "images" / path_str
    if asset_path.exists(): return asset_path
    raise FileNotFoundError(f"Image not found: {path_str}")


def save_fallback_visualizations(points: np.ndarray, colors: np.ndarray, save_dir: Path) -> None:
    """Save static 3D PLY and 2D Scatter if window fails."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save 3D PLY (Colorized)
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        ply_path = save_dir / "sanity_ghosting.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"Saved 3D cloud: {ply_path}")

    # 2. Save 2D Heatmap Diagram (Matplotlib)
    if len(points) == 0: return
    
    plt.figure(figsize=(8, 8))
    # Scatter plot X vs Y (Top-down view)
    # colors array is [R, G, B]. We can just pass it directly.
    plt.scatter(points[:, 0], points[:, 1], c=colors, s=10, alpha=0.8)
    
    plt.title(f"Ghosting Variance (Red=High, Blue=Low)")
    plt.xlabel("X (World)")
    plt.ylabel("Y (World)")
    plt.axis('equal')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    png_path = save_dir / "sanity_ghosting.png"
    plt.savefig(png_path)
    plt.close()
    print(f"Saved 2D heatmap: {png_path}")


def main():
    args = parse_args()
    
    try:
        img_path = resolve_image_path(args.image)
    except FileNotFoundError as e:
        print(e)
        return
        
    print(f"--- Ghosting Sanity Check ---")
    print(f"Loading {img_path.name}...")
    
    # 1. Configure
    cfg = ActiveHallucinationConfig()
    cfg.pointe.num_seeds = args.seeds
    cfg.variance.resolution = args.resolution
    
    # 2. Generate Point Clouds
    gen = PointEGenerator(cfg.pointe)
    img = np.array(Image.open(img_path).convert("RGB"))
    clouds = gen.generate_point_clouds_from_image(img, prompt="object")

    # 3. Compute Variance Field (Using your functional API)
    print("Computing Variance Field...")
    var_grid = compute_variance_field(clouds, cfg.variance)
    
    # 4. Statistics
    max_var = np.max(var_grid)
    mean_var = np.mean(var_grid)
    high_var_count = np.sum(var_grid > 0.15)
    
    print(f"\n--- Stats ---")
    print(f"Max Variance:  {max_var:.4f} (Theoretical max ~0.25)")
    print(f"Mean Variance: {mean_var:.4f}")
    print(f"Ghost Voxels:  {high_var_count} (Var > 0.15)")

    if max_var == 0:
        print("WARNING: Variance is 0. Check random seeds or input image.")
    else:
        print("SUCCESS: Variance detected.")

    # 5. Prepare Visualization
    # Use the VoxelGrid class helper from your file to map back to world coords
    grid_helper = VoxelGrid(bounds=cfg.variance.grid_bounds, resolution=cfg.variance.resolution)
    
    # Export points with variance > 0.01
    score_points = export_score_points(var_grid, grid_helper, threshold=0.01)
    
    if len(score_points) == 0:
        print("No variance voxels to visualize.")
        return

    xyz = score_points[:, :3]
    variances = score_points[:, 3]
    
    # Color Mapping
    # Normalize variance: 0.0 -> 0.25 maps to 0 -> 1
    norm_v = np.clip(variances / 0.25, 0, 1)
    colors = np.zeros((len(variances), 3))
    colors[:, 0] = norm_v       # Red channel = High Variance
    colors[:, 2] = 1.0 - norm_v # Blue channel = Low Variance

    # 6. Visualize
    print("Visualizing...")
    try:
        if o3d is None: raise ImportError("Open3D not available")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        
        print("Attempting interactive window...")
        o3d.visualization.draw_geometries([pcd, axes], window_name="Ghosting Check")
        
    except Exception as exc:
        print(f"Window unavailable ({exc}). Saving fallbacks...")
        save_fallback_visualizations(xyz, colors, PROJECT_ROOT / "outputs")


if __name__ == "__main__":
    main()