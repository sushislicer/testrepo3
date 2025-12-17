"""
Sanity check using the implemented VarianceField class.
Usage: python src/scripts/sanity_check_ghosting.py --image assets/images/chair.png
"""

from __future__ import annotations

import argparse
import sys
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
from src.variance_field import SemanticVarianceField

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image.")
    parser.add_argument("--seeds", type=int, default=5, help="Number of Point-E seeds.")
    parser.add_argument("--resolution", type=int, default=64, help="Voxel grid resolution.")
    return parser.parse_args()

def main():
    args = parse_args()
    img_path = Path(args.image)
    if not img_path.exists():
        img_path = PROJECT_ROOT / "assets" / "images" / args.image
    print(f"--- Ghosting Sanity Check (Using VarianceField) ---")
    print(f"Generating {args.seeds} clouds...")
    cfg = ActiveHallucinationConfig()
    cfg.pointe.num_seeds = args.seeds
    
    gen = PointEGenerator(cfg.pointe)
    img = np.array(Image.open(img_path).convert("RGB"))
    clouds = gen.generate_point_clouds_from_image(img, prompt="object")

    print("Computing Variance Field...")
    field = SemanticVarianceField(resolution=args.resolution, radius=0.5)
    field.accumulate_point_clouds(clouds)
    field.compute_