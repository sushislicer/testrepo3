"""Compute and visualize the variance field for a single RGB input."""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
from PIL import Image

from active_hallucination.config import ActiveHallucinationConfig
from active_hallucination.pointe_wrapper import PointEGenerator
from active_hallucination.segmentation import CLIPSegSegmenter
from active_hallucination.variance_field import (
    VoxelGrid,
    compute_variance_field,
    accumulate_semantic_weights,
    combine_variance_and_semantics,
    export_score_points,
)
from active_hallucination.visualization import save_variance_max_projection
from active_hallucination.simulator import pose_to_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Variance field demo from a single RGB image.")
    parser.add_argument("--image", type=str, required=True, help="Path to RGB image.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt for Point-E.")
    parser.add_argument("--save_dir", type=str, default="outputs/demo_variance", help="Directory for outputs.")
    parser.add_argument("--use_mask", action="store_true", help="Apply CLIPSeg semantic weighting if available.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ActiveHallucinationConfig()
    if args.prompt:
        cfg.pointe.prompt = args.prompt
    img = np.array(Image.open(args.image).convert("RGB"))

    pointe = PointEGenerator(cfg.pointe)
    seg = CLIPSegSegmenter(cfg.segmentation)
    clouds = pointe.generate_point_clouds_from_image(img, prompt=cfg.pointe.prompt, num_seeds=cfg.pointe.num_seeds)
    variance = compute_variance_field(clouds, cfg.variance)

    semantic_grid = np.ones_like(variance)
    if args.use_mask:
        mask = seg.segment_affordance(img, cfg.segmentation.prompt)
        # Use identity pose; assumes points already in camera frame. This is a simplification.
        identity_pose = np.eye(4, dtype=np.float32)
        point_weights = [
            np.ones(pc.shape[0], dtype=np.float32)  # default uniform
            for pc in clouds
        ]
        try:
            from active_hallucination.segmentation import compute_point_mask_weights

            point_weights = [
                compute_point_mask_weights(pc, identity_pose, cfg.simulator.intrinsics, mask) for pc in clouds
            ]
        except Exception:
            pass
        semantic_grid = accumulate_semantic_weights(clouds, point_weights, cfg.variance)

    score = combine_variance_and_semantics(variance, semantic_grid)
    out_dir = Path(args.save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_variance_max_projection(score, out_dir / "variance_projection.png")
    grid = VoxelGrid(bounds=cfg.variance.grid_bounds, resolution=cfg.variance.resolution)
    points = export_score_points(score, grid, threshold=0.3)
    np.save(out_dir / "variance_points.npy", points)
    print(f"Saved variance projection and points to {out_dir}")


if __name__ == "__main__":
    main()
