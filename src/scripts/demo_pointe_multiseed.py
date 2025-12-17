"""Generate multiple Point-E point clouds from a single RGB image."""

from __future__ import annotations

import argparse
import sys
import colorsys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import ActiveHallucinationConfig
from src.pointe_wrapper import PointEGenerator
from src.visualization import overlay_point_clouds_open3d

try:
    import open3d as o3d
except ImportError:
    o3d = None

DEFAULT_IMAGE_DIR = PROJECT_ROOT / "assets" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-E multi-seed sampling demo.")
    parser.add_argument("--image", type=str, required=True, help=f"Filename or path.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt.")
    parser.add_argument("--num_seeds", type=int, default=None, help="Number of samples.")
    parser.add_argument("--save_dir", type=str, default="outputs/demo_pointe", help="Output directory.")
    parser.add_argument("--view", action="store_true", help="Open Open3D overlay if available.")
    return parser.parse_args()


def resolve_image_path(input_path_str: str) -> Path:
    path = Path(input_path_str)
    if path.exists() and path.is_file():
        return path
    asset_path = DEFAULT_IMAGE_DIR / input_path_str
    if asset_path.exists() and asset_path.is_file():
        return asset_path
    raise FileNotFoundError(f"Could not find image: {input_path_str}")


def save_fallback_visualizations(point_clouds: list[np.ndarray], save_dir: Path) -> None:
    if not point_clouds: return

    # Save Merged PLY
    if o3d is not None:
        merged_pcd = o3d.geometry.PointCloud()
        all_points, all_colors = [], []
        for idx, pc in enumerate(point_clouds):
            if len(pc) == 0: continue
            hue = idx / max(len(point_clouds), 1)
            rgb = colorsys.hsv_to_rgb(hue, 0.7, 1.0)
            all_points.append(pc)
            all_colors.append(np.tile(rgb, (pc.shape[0], 1)))
            
        if all_points:
            merged_pcd.points = o3d.utility.Vector3dVector(np.vstack(all_points))
            merged_pcd.colors = o3d.utility.Vector3dVector(np.vstack(all_colors))
            ply_path = save_dir / "multiseed_merged.ply"
            o3d.io.write_point_cloud(str(ply_path), merged_pcd)
            print(f"Saved merged 3D cloud to: {ply_path}")

    # Save 2D Scatter Plot
    plt.figure(figsize=(8, 8))
    for idx, pc in enumerate(point_clouds):
        if len(pc) == 0: continue
        plt.scatter(pc[:, 0], pc[:, 1], s=1, alpha=0.5, label=f"Seed {idx}")
    plt.title(f"Multi-seed Scatter (Top-down)")
    plt.legend()
    plt.axis('equal')
    png_path = save_dir / "multiseed_overlay.png"
    plt.savefig(png_path)
    plt.close()
    print(f"Saved 2D visualization to: {png_path}")


def main() -> None:
    args = parse_args()

    try:
        img_path = resolve_image_path(args.image)
        print(f"Loading image from: {img_path}")
    except FileNotFoundError as e:
        print(e)
        return

    cfg = ActiveHallucinationConfig()
    if args.prompt:
        cfg.pointe.prompt = args.prompt
    if args.num_seeds:
        cfg.pointe.num_seeds = args.num_seeds

    img = np.array(Image.open(img_path).convert("RGB"))
    gen = PointEGenerator(cfg.pointe)
    pcs = gen.generate_point_clouds_from_image(img, prompt=cfg.pointe.prompt, num_seeds=cfg.pointe.num_seeds)

    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i, pc in enumerate(pcs):
        np.save(out_dir / f"cloud_{i:02d}.npy", pc)
    print(f"Saved {len(pcs)} point clouds to {out_dir}")

    if args.view:
        try:
            print("Attempting interactive window...")
            overlay_point_clouds_open3d(pcs)
        except Exception as exc:
            print(f"Window unavailable ({exc}). Saving fallbacks...")
            save_fallback_visualizations(pcs, out_dir)


if __name__ == "__main__":
    main()