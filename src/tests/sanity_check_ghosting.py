"""Sanity check: Ghosting artifacts via variance field."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

try:
    import open3d as o3d
except ImportError:
    o3d = None

from src.config import ActiveHallucinationConfig
from src.pointe_wrapper import PointEGenerator
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


def save_visualizations(points: np.ndarray, colors: np.ndarray, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        ply_path = save_dir / "sanity_ghosting.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"Saved 3D cloud: {ply_path}")

    if len(points) > 0:
        plt.figure(figsize=(8, 8))
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
        
    print(f"Loading {img_path.name}...")
    cfg = ActiveHallucinationConfig()
    cfg.pointe.num_seeds = args.seeds
    cfg.variance.resolution = args.resolution
    
    gen = PointEGenerator(cfg.pointe)
    img = np.array(Image.open(img_path).convert("RGB"))
    clouds = gen.generate_point_clouds_from_image(img, prompt="object")

    print("Computing Variance Field...")
    var_grid = compute_variance_field(clouds, cfg.variance)
    
    max_var = np.max(var_grid)
    mean_var = np.mean(var_grid)
    high_var_count = np.sum(var_grid > 0.15)
    
    print(f"Max Variance: {max_var:.4f}")
    print(f"Mean Variance: {mean_var:.4f}")
    print(f"Ghost Voxels: {high_var_count}")

    if max_var == 0:
        print("WARNING: Variance is 0. Check random seeds.")

    grid_helper = VoxelGrid(bounds=cfg.variance.grid_bounds, resolution=cfg.variance.resolution)
    score_points = export_score_points(var_grid, grid_helper, threshold=0.01)
    
    if len(score_points) == 0:
        print("No variance voxels to visualize.")
        return

    xyz = score_points[:, :3]
    variances = score_points[:, 3]
    
    norm_v = np.clip(variances / 0.25, 0, 1)
    colors = np.zeros((len(variances), 3))
    colors[:, 0] = norm_v       
    colors[:, 2] = 1.0 - norm_v 

    print("Saving outputs...")
    save_visualizations(xyz, colors, PROJECT_ROOT / "outputs")

    if o3d is not None:
        try:
            print("Attempting interactive window...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([pcd, axes], window_name="Ghosting Check")
        except Exception as exc:
            print(f"Interactive window skipped ({exc}). View saved files.")
    else:
        print("Open3D not installed. Window skipped.")


if __name__ == "__main__":
    main()