"""
Sanity Check: Semantic Variance Field Integration.

This script verifies the full pipeline:
1. Image -> Point-E (3D Clouds)
2. Image -> CLIPSeg (2D Mask)
3. 2D Mask + 3D Clouds -> Projection -> Semantic Voxel Grid
4. Semantic Grid * Variance Grid -> Final Score Field

Usage:
    python src/scripts/sanity_check_semantic.py --image assets/images/mug.png --prompt "handle"
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

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
    """
    Creates a virtual camera positioned to view the Point-E object from the front.
    Point-E usually generates objects centered at (0,0,0) within a unit cube.
    """
    # FOV ~60 degrees is standard. fx = width / (2 * tan(fov/2))
    focal = image_size  # Approx 53 degrees FOV
    intrinsics = SimpleIntrinsics(
        fx=focal, fy=focal, cx=image_size / 2, cy=image_size / 2
    )

    # Place camera at (0, 0, 2.0) looking at (0, 0, 0)
    # Assuming Pyrender/OpenGL convention where camera looks down -Z
    cam_to_world = np.eye(4, dtype=np.float32)
    cam_to_world[:3, 3] = [0.0, 0.0, 2.0]
    
    return cam_to_world, intrinsics

def main():
    parser = argparse.ArgumentParser(description="Test Semantic Variance Field.")
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--prompt", type=str, default="handle", help="Text prompt for CLIPSeg.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of Point-E clouds to generate.")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        img_path = PROJECT_ROOT / args.image
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
    print(f"Max Variance: {var_grid.max():.4f}")
    print(f"Max Semantic: {sem_grid.max():.4f}")
    print(f"Max Combined: {score_grid.max():.4f}")
    
    if score_grid.max() == 0:
        print("[WARNING] Combined score is zero. Check if projection aligned points with the mask.")
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
    
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    
    # Colormap: Purple (Low) -> Yellow (High)
    colors = np.zeros((len(scores), 3))
    colors[:, 0] = scores          # Red
    colors[:, 1] = scores * 0.8    # Green
    colors[:, 2] = 1.0 - scores    # Blue
    pcd.colors = o3d.utility.Vector3dVector(colors)
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    
    try:
        o3d.visualization.draw_geometries([pcd, axes], window_name=f"Semantic: {args.prompt}")
    except Exception:
        out_path = PROJECT_ROOT / "outputs" / "semantic_check.ply"
        o3d.io.write_point_cloud(str(out_path), pcd)
        print(f"Headless environment detected. Saved PLY to {out_path}")

if __name__ == "__main__":
    main()