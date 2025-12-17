"""Compute and visualize the variance field for a single RGB input."""

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

DEFAULT_IMAGE_DIR = PROJECT_ROOT / "assets" / "images"

from src.config import ActiveHallucinationConfig
from src.pointe_wrapper import PointEGenerator
from src.segmentation import CLIPSegSegmenter, compute_point_mask_weights
from src.variance_field import (
    VoxelGrid,
    compute_variance_field,
    accumulate_semantic_weights,
    combine_variance_and_semantics,
    export_score_points,
)
from src.visualization import save_variance_max_projection
from src.simulator import pose_to_matrix

try:
    import open3d as o3d
except ImportError:
    o3d = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Variance field demo from a single RGB image.")
    parser.add_argument("--image", type=str, required=True, help="Path to RGB image or filename.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for Point-E.")
    parser.add_argument("--save_dir", type=str, default="outputs/demo_variance", help="Output directory.")
    parser.add_argument("--use_mask", action="store_true", help="Apply CLIPSeg semantic weighting.")
    parser.add_argument("--view", action="store_true", help="Attempt interactive 3D visualization.")
    return parser.parse_args()


def resolve_image_path(name_or_path: str) -> Path:
    path = Path(name_or_path)
    if path.exists(): return path
    asset_path = DEFAULT_IMAGE_DIR / name_or_path
    if asset_path.exists(): return asset_path
    raise FileNotFoundError(f"Image not found: {name_or_path}")


def save_fallback_diagrams(points: np.ndarray, save_dir: Path) -> None:
    """Save 3D PLY and 2D Scatter plot when window fails."""
    if len(points) == 0: return
    
    xyz = points[:, :3]
    scores = points[:, 3]

    # 1. Save PLY
    if o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)
        # Color map: Plasma
        cmap = plt.get_cmap("plasma")
        colors = cmap(scores)[:, :3]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        ply_path = save_dir / "variance_field.ply"
        o3d.io.write_point_cloud(str(ply_path), pcd)
        print(f"Saved 3D PLY to: {ply_path}")

    # 2. Save 2D Diagram
    plt.figure(figsize=(8, 8))
    plt.scatter(xyz[:, 0], xyz[:, 1], c=scores, cmap="plasma", s=10, alpha=0.8)
    plt.colorbar(label="Variance/Score")
    plt.title("Variance Field (Top-Down)")
    plt.axis('equal')
    png_path = save_dir / "variance_scatter.png"
    plt.savefig(png_path)
    plt.close()
    print(f"Saved 2D diagram to: {png_path}")


def main() -> None:
    args = parse_args()
    
    try:
        img_path = resolve_image_path(args.image)
    except FileNotFoundError as e:
        print(e)
        return

    cfg = ActiveHallucinationConfig()
    if args.prompt:
        cfg.pointe.prompt = args.prompt
    
    print(f"Loading {img_path}...")
    img = np.array(Image.open(img_path).convert("RGB"))

    pointe = PointEGenerator(cfg.pointe)
    seg = CLIPSegSegmenter(cfg.segmentation)
    
    clouds = pointe.generate_point_clouds_from_image(img, prompt=cfg.pointe.prompt, num_seeds=cfg.pointe.num_seeds)
    variance = compute_variance_field(clouds, cfg.variance)

    semantic_grid = np.ones_like(variance)
    if args.use_mask:
        mask = seg.segment_affordance(img, cfg.segmentation.prompt)
        identity_pose = np.eye(4, dtype=np.float32)
        try:
            point_weights = [
                compute_point_mask_weights(pc, identity_pose, cfg.simulator.intrinsics, mask) 
                for pc in clouds
            ]
            semantic_grid = accumulate_semantic_weights(clouds, point_weights, cfg.variance)
        except Exception:
            pass

    score = combine_variance_and_semantics(variance, semantic_grid)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    save_variance_max_projection(score, out_dir / "variance_projection.png")
    
    grid = VoxelGrid(bounds=cfg.variance.grid_bounds, resolution=cfg.variance.resolution)
    points = export_score_points(score, grid, threshold=0.3)
    np.save(out_dir / "variance_points.npy", points)
    print(f"Saved variance projection and points to {out_dir}")

    # Visualization Logic
    if args.view and len(points) > 0 and o3d is not None:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # Simple color mapping for view
        scores = points[:, 3]
        colors = np.zeros((len(scores), 3))
        colors[:, 0] = scores; colors[:, 1] = scores