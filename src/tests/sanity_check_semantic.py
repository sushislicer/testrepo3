"""
Sanity Check: Semantic Variance Field Integration.
Location: src/tests/sanity_check_semantic.py
Usage: python src/tests/sanity_check_semantic.py --image assets/images/chair.png --prompt "leg" --seeds 10
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

# --- Headless Import Safety ---
try:
    import open3d as o3d
except ImportError:
    o3d = None

# --- Internal Imports ---
from src.config import ActiveHallucinationConfig
from src.pointe_wrapper import PointEGenerator
from src.segmentation import CLIPSegSegmenter, compute_point_mask_weights
from src.variance_field import (
    compute_variance_field,
    accumulate_semantic_weights,
    combine_variance_and_semantics,
    export_score_points,
    VoxelGrid
)


@dataclass
class SimpleIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float


def get_virtual_camera(image_size: int = 512):
    focal = image_size
    intrinsics = SimpleIntrinsics(
        fx=focal, fy=focal, cx=image_size / 2, cy=image_size / 2
    )
    cam_to_world = np.eye(4, dtype=np.float32)
    cam_to_world[:3, 3] = [0.0, 0.0, 2.0]
    return cam_to_world, intrinsics


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists(): return path
    asset_path = PROJECT_ROOT / "assets" / "images" / path_str
    if asset_path.exists(): return asset_path
    root_path = PROJECT_ROOT / path_str
    if root_path.exists(): return root_path
    raise FileNotFoundError(f"Image not found: {path_str}")


def save_fallback_visualizations(pcd_points: np.ndarray, pcd_colors: np.ndarray, save_dir: Path, prompt: str) -> None:
    """Saves static files if Open3D window cannot open."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Save 3D PLY
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pcd_points)
        pcd.colors = o3d.utility.Vector3dVector(pcd_colors)
        ply_path = save_dir / "sanity_semantic.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"Saved 3D cloud: {ply_path}")

    # 2. Save 2D Scatter Diagram
    if len(pcd_points) == 0: return

    plt.figure(figsize=(8, 8))
    # Plot X vs Y (Top down)
    plt.scatter(pcd_points[:, 0], pcd_points[:, 1], c=pcd_colors, s=10, alpha=0.8)
    plt.title(f"Semantic Score: '{prompt}' (Yellow=High)")
    plt.axis('equal')
    png_path = save_dir / "sanity_semantic.png"
    plt.savefig(png_path)
    plt.close()
    print(f"Saved 2D diagram: {png_path}")


def main():
    parser = argparse.ArgumentParser(description="Test Semantic Variance Field.")
    parser.add_argument("--image", type=str, required=True, help="Input image.")
    parser.add_argument("--prompt", type=str, default="handle", help="Text prompt.")
    parser.add_argument("--seeds", type=int, default=5, help="Point-E seeds.")
    args = parser.parse_args()

    try:
        img_path = resolve_image_path(args.image)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"--- Semantic Variance Sanity Check ---")
    print(f"Target: '{args.prompt}' in {img_path.name}")
    pil_img = Image.open(img_path).convert("RGB").resize((512, 512))
    img_arr = np.array(pil_img)

    print(f"1. Generating {args.seeds} point clouds...")
    cfg = ActiveHallucinationConfig()
    cfg.pointe.num_seeds = args.seeds
    gen = PointEGenerator(cfg.pointe)
    clouds = gen.generate_point_clouds_from_image(img_arr)
    
    print(f"2. Computing 2D Affordance Mask...")
    seg = CLIPSegSegmenter(cfg.segmentation)
    mask = seg.segment_affordance(img_arr, prompt=args.prompt)
    
    mask_debug_path = PROJECT_ROOT / "outputs" / "debug_mask.png"
    mask_debug_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_debug_path)
    print(f"   -> Mask saved to {mask_debug_path}")

    print(f"3. Projecting 2D mask to 3D points...")
    cam_to_world, intrinsics = get_virtual_camera(image_size=512)
    point_weights_list = []
    for cloud in clouds:
        w = compute_point_mask_weights(cloud, cam_to_world, intrinsics, mask)
        point_weights_list.append(w)

    print(f"4. Computing Voxel Fields (Resolution: {cfg.variance.resolution})...")
    var_grid = compute_variance_field(clouds, cfg.variance)
    sem_grid = accumulate_semantic_weights(clouds, point_weights_list, cfg.variance)
    score_grid = combine_variance_and_semantics(var_grid, sem_grid)

    print(f"\n--- Results ---")
    print(f"Max Combined: {score_grid.max():.4f}")
    
    if score_grid.max() == 0:
        print("[WARNING] Combined score is zero. Check projection alignment.")
    else:
        print("[SUCCESS] High-scoring regions identified.")

    print("\n5. Visualizing High-Score Voxels...")
    grid_obj = VoxelGrid(bounds=cfg.variance.grid_bounds, resolution=cfg.variance.resolution)
    score_points = export_score_points(score_grid, grid_obj, threshold=0.01)
    
    if len(score_points) == 0:
        print("No voxels passed the threshold.")
        return

    xyz = score_points[:, :3]
    scores = score_points[:, 3]
    
    # Prepare Data for Visualization
    # Colormap: Purple (Low) -> Yellow (High)
    colors = np.zeros((len(scores), 3))
    colors[:, 0] = scores          # Red
    colors[:, 1] = scores * 0.8    # Green
    colors[:, 2] = 1.0 - scores    # Blue

    # Attempt Interactive Window
    try:
        if o3d is None: raise ImportError("Open3D not available")
        print("Attempting interactive window...")
        
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        o3d.visualization.draw_geometries([pcd, axes], window_name=f"Semantic: {args.prompt}")
        
    except Exception as exc:
        print(f"Window unavailable ({exc}). Saving fallbacks...")
        save_fallback_visualizations(xyz, colors, PROJECT_ROOT / "outputs", args.prompt)


if __name__ == "__main__":
    main()