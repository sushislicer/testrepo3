"""
Sanity Check: Semantic Variance Field Integration.
Location: src/tests/sanity_check_semantic.py
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

try:
    import open3d as o3d
except ImportError:
    o3d = None

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
    fx: float; fy: float; cx: float; cy: float


def get_virtual_camera(image_size: int = 512):
    focal = image_size
    intrinsics = SimpleIntrinsics(fx=focal, fy=focal, cx=image_size/2, cy=image_size/2)
    # Camera at (0, 0, 2.5) looking at (0, 0, 0)
    cam_to_world = np.eye(4, dtype=np.float32)
    cam_to_world[:3, 3] = [0.0, 0.0, 2.5]
    return cam_to_world, intrinsics


def resolve_image_path(path_str: str) -> Path:
    path = Path(path_str)
    if path.exists(): return path
    asset_path = PROJECT_ROOT / "assets" / "images" / path_str
    if asset_path.exists(): return asset_path
    root_path = PROJECT_ROOT / path_str
    if root_path.exists(): return root_path
    raise FileNotFoundError(f"Image not found: {path_str}")


def save_projection_debug(cloud: np.ndarray, cam_to_world: np.ndarray, intr, mask: np.ndarray, save_dir: Path):
    """
    Crucial Debugger: Plots the 2D mask and overlays the projected 3D points.
    If points are missing or off-center, the camera/projection logic is wrong.
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Project Points manually for visualization
    world_to_cam = np.linalg.inv(cam_to_world)
    ones = np.ones((cloud.shape[0], 1))
    hom = np.hstack([cloud, ones])
    cam_pts = (world_to_cam @ hom.T).T
    
    # Standard Pinhole Projection
    # Note: Check coordinate system. Usually -Z is forward in CV, but Point-E might vary.
    # We assume X=Right, Y=Up, Z=Forward/Back
    x, y, z = cam_pts[:, 0], cam_pts[:, 1], cam_pts[:, 2]
    
    # Filter points behind camera
    valid_z = np.abs(z) > 0.1
    
    # Simple projection u = fx * x/z + cx
    # Note: If Z is negative (standard OpenGL), use -z. If positive, use z.
    # Trying standard division:
    u = (intr.fx * x / -z) + intr.cx
    v = (intr.fy * y / -z) + intr.cy # Flip Y if image is inverted
    
    u = u[valid_z]
    v = v[valid_z]

    # 2. Plot
    plt.figure(figsize=(10, 10))
    plt.imshow(mask, cmap='gray', origin='upper') # Mask background
    
    # Plot projected points
    # We subsample points to keep plot clean
    if len(u) > 0:
        indices = np.random.choice(len(u), min(len(u), 2000), replace=False)
        plt.scatter(u[indices], v[indices], s=2, c='red', alpha=0.5, label='Projected 3D Points')
    else:
        print("[DEBUG] No valid points in front of camera to project!")

    plt.legend()
    plt.title("DEBUG: 2D Mask (Gray) + Projected 3D Points (Red)\nRed dots MUST overlap white areas.")
    plt.xlim(0, mask.shape[1])
    plt.ylim(mask.shape[0], 0) # Flip Y to match image coords
    
    out_path = save_dir / "debug_projection_alignment.png"
    plt.savefig(out_path)
    plt.close()
    print(f"Saved projection debug image to: {out_path}")


def save_visualizations(points: np.ndarray, colors: np.ndarray, save_dir: Path, prompt: str) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        ply_path = save_dir / "sanity_semantic.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"Saved 3D cloud: {ply_path}")

    if len(points) > 0:
        plt.figure(figsize=(8, 8))
        plt.scatter(points[:, 0], points[:, 1], c=colors, s=10, alpha=0.8)
        plt.title(f"Semantic Score: '{prompt}' (Yellow=High)")
        plt.axis('equal')
        png_path = save_dir / "sanity_semantic.png"
        plt.savefig(png_path)
        plt.close()
        print(f"Saved 2D diagram: {png_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Input image.")
    parser.add_argument("--prompt", type=str, default="handle", help="Text prompt.")
    parser.add_argument("--seeds", type=int, default=5, help="Point-E seeds.")
    args = parser.parse_args()

    try:
        img_path = resolve_image_path(args.image)
    except FileNotFoundError as e:
        print(e)
        return

    print(f"--- Semantic Check: '{args.prompt}' ---")
    pil_img = Image.open(img_path).convert("RGB").resize((512, 512))
    img_arr = np.array(pil_img)

    cfg = ActiveHallucinationConfig()
    cfg.pointe.num_seeds = args.seeds
    
    # 1. Point-E
    clouds = PointEGenerator(cfg.pointe).generate_point_clouds_from_image(img_arr)
    
    # 2. CLIPSeg
    mask = CLIPSegSegmenter(cfg.segmentation).segment_affordance(img_arr, prompt=args.prompt)
    
    # Mask Diagnostics
    mask_max = mask.max()
    print(f"[DEBUG] Mask Max Value: {mask_max:.4f}")
    if mask_max < 0.01:
        print("  -> ERROR: CLIPSeg found nothing. Check your prompt or image.")
    
    mask_path = PROJECT_ROOT / "outputs" / "debug_mask.png"
    mask_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((mask * 255).astype(np.uint8)).save(mask_path)

    # 3. Projection & Weights
    cam_to_world, intrinsics = get_virtual_camera(512)
    
    # --- RUN NEW PROJECTION DEBUGGER ---
    if clouds:
        save_projection_debug(clouds[0], cam_to_world, intrinsics, mask, PROJECT_ROOT / "outputs")
    
    point_weights_list = [compute_point_mask_weights(c, cam_to_world, intrinsics, mask) for c in clouds]
    
    # Weights Diagnostics
    total_weight = sum([w.sum() for w in point_weights_list])
    print(f"[DEBUG] Total Semantic Weight accumulated: {total_weight:.4f}")
    if total_weight == 0:
        print("  -> ERROR: No 3D points landed on the 2D mask. Check 'debug_projection_alignment.png'.")

    # 4. Fields
    var_grid = compute_variance_field(clouds, cfg.variance)
    sem_grid = accumulate_semantic_weights(clouds, point_weights_list, cfg.variance)
    score_grid = combine_variance_and_semantics(var_grid, sem_grid)

    print(f"\n[Stats] Max Variance: {var_grid.max():.4f}")
    print(f"[Stats] Max Semantic Grid: {sem_grid.max():.4f}")
    print(f"[Stats] Max Combined Score: {score_grid.max():.4f}")

    grid_obj = VoxelGrid(bounds=cfg.variance.grid_bounds, resolution=cfg.variance.resolution)
    # Lower threshold slightly to see faint signals
    score_points = export_score_points(score_grid, grid_obj, threshold=0.001)
    
    if len(score_points) == 0:
        print("No voxels above threshold.")
        return

    xyz = score_points[:, :3]
    scores = score_points[:, 3]
    
    colors = np.zeros((len(scores), 3))
    colors[:, 0] = scores          
    colors[:, 1] = scores * 0.8    
    colors[:, 2] = 1.0 - scores    

    print("Saving outputs...")
    save_visualizations(xyz, colors, PROJECT_ROOT / "outputs", args.prompt)

    if o3d is not None:
        try:
            print("Attempting interactive window...")
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(xyz)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
            o3d.visualization.draw_geometries([pcd, axes], window_name=f"Semantic: {args.prompt}")
        except Exception as exc:
            print(f"Interactive window skipped ({exc}). View saved files.")
    else:
        print("Open3D missing. Window skipped.")


if __name__ == "__main__":
    main()