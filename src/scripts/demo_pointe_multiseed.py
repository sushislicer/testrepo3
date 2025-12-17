"""Generate multiple Point-E point clouds from a single RGB image."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
from PIL import Image

# Add the project root to sys.path so we can import from src
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import ActiveHallucinationConfig
from src.pointe_wrapper import PointEGenerator
from src.visualization import overlay_point_clouds_open3d

DEFAULT_IMAGE_DIR = PROJECT_ROOT / "assets" / "images"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Point-E multi-seed sampling demo.")
    parser.add_argument(
        "--image", 
        type=str, 
        required=True, 
        help=f"Filename of image in {DEFAULT_IMAGE_DIR} OR full path to input RGB image."
    )
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for Point-E guidance.")
    parser.add_argument("--num_seeds", type=int, default=None, help="Number of samples.")
    parser.add_argument("--save_dir", type=str, default="outputs/demo_pointe", help="Directory to save npy outputs.")
    parser.add_argument("--view", action="store_true", help="Open interactive Open3D overlay if available.")
    return parser.parse_args()


def resolve_image_path(input_path_str: str) -> Path:
    path = Path(input_path_str)
    if path.exists() and path.is_file():
        return path
    asset_path = DEFAULT_IMAGE_DIR / input_path_str
    if asset_path.exists() and asset_path.is_file():
        return asset_path
    raise FileNotFoundError(
        f"Could not find image: '{input_path_str}'.\n"
        f"Checked CWD and default asset directory: {DEFAULT_IMAGE_DIR}"
    )


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
            overlay_point_clouds_open3d(pcs)
        except Exception as exc:
            print(f"Open3D visualization unavailable: {exc}")


if __name__ == "__main__":
    main()