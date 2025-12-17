"""
Sanity check: Ghosting artifacts via variance field.
Usage: python src/tests/sanity_check_ghosting.py --image assets/images/chair.png --resolution 64
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
from PIL import Image

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import ActiveHallucinationConfig
from src.pointe_wrapper import PointEGenerator
from src.variance_field import SemanticVarianceField


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
    root_path = PROJECT_ROOT / path_str
    if root_path.exists(): return root_path
    raise FileNotFoundError(f"Image not found: {path_str}")


def save_fallback_visualizations(pcd: o3d.geometry.PointCloud, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Save 3D PLY
    ply_path = save_dir / "sanity_ghosting.ply"
    o3d.io.write_point_cloud(str(ply_path), pcd)
    print(f"Saved 3D cloud: {ply_path}")

    # Save 2D Heatmap Diagram
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    if len(points) == 0: return

    plt.figure(figsize=(8, 8))
    # Top-down projection (X vs Y)
    plt.scatter(points[:, 0], points[:, 1], c=colors, s=10, alpha=0.8)
    plt.title("Variance Heatmap (Red=High, Blue=Low)")
    plt.axis('equal')
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
    
    cfg = ActiveHallucinationConfig()
    cfg.pointe.num_seeds = args.seeds
    
    gen = PointEGenerator(cfg.pointe)
    img = np.array(Image.open(img_path).convert("RGB"))
    clouds = gen.generate_point_clouds_from_image(img, prompt="object")

    print("Computing Variance Field...")
    field = SemanticVarianceField(resolution=args.resolution, radius=0.5)
    field.accumulate_point_clouds(clouds)
    field.compute_variance_field()
    
    stats = field.get_statistics()
    print(f"\n--- Stats ---")
    print(f"Max Variance:  {stats['max']:.4f}")
    print(f"Mean Variance: {stats['mean']:.4f}")
    print(f"Ghost Voxels:  {stats['high_var_count']} (Var > 0.15)")

    if stats['max'] == 0:
        print("WARNING: Variance is 0. Check random seeds.")
    else:
        print("SUCCESS: Variance detected.")

    pcd = field.export_debug_cloud(color_by_variance=True)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    try:
        print("Attempting interactive window...")
        o3d.visualization.draw_geometries([pcd, axes], window_name="Ghosting Check")
    except Exception as exc:
        print(f"Window unavailable ({exc}). Saving fallbacks...")
        save_fallback_visualizations(pcd, PROJECT_ROOT / "outputs")


if __name__ == "__main__":
    main()