"""Experiment orchestration for Active-Hallucination."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from .config import ActiveHallucinationConfig
from .pointe_wrapper import PointEGenerator
from .segmentation import CLIPSegSegmenter, compute_point_mask_weights
from .simulator import VirtualTabletopSimulator, pose_to_matrix
from .variance_field import (
    VoxelGrid,
    compute_variance_field,
    accumulate_semantic_weights,
    combine_variance_and_semantics,
    extract_topk_centroid,
)
from .nbv_policy import (
    select_next_view_active,
    select_next_view_random,
    select_next_view_geometric,
)

logger = logging.getLogger(__name__)


@dataclass
class StepLog:
    step: int
    view_id: int
    next_view_id: int
    variance_sum: float
    policy_score: float
    centroid: List[float]


def _ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


class ActiveHallucinationRunner:
    """Runs the Active-Hallucination loop for a single object."""

    def __init__(self, cfg: ActiveHallucinationConfig):
        self.cfg = cfg
        self.sim = VirtualTabletopSimulator(cfg.simulator)
        self.pointe = PointEGenerator(cfg.pointe)
        self.segmenter = CLIPSegSegmenter(cfg.segmentation)
        self.rng = np.random.RandomState(cfg.policy.random_seed)
        self._output_root = Path(cfg.experiment.output_dir) / cfg.experiment.trajectory_name
        _ensure_dir(self._output_root)

    def _save_debug(self, step: int, rgb: np.ndarray, mask: np.ndarray, score_grid: np.ndarray) -> None:
        if not self.cfg.experiment.save_debug:
            return
        import imageio

        imageio.imwrite(self._output_root / f"step_{step:02d}_rgb.png", rgb.astype(np.uint8))
        imageio.imwrite(self._output_root / f"step_{step:02d}_mask.png", (mask * 255).astype(np.uint8))
        # Save max-projection of score grid for quick inspection
        proj = np.max(score_grid, axis=2)
        proj = (proj / (proj.max() + 1e-8) * 255).astype(np.uint8)
        imageio.imwrite(self._output_root / f"step_{step:02d}_score.png", proj)

    def run_episode(self, initial_view: int = 0) -> Dict:
        cfg = self.cfg
        poses = self.sim.list_poses()
        visited: Set[int] = set()
        logs: List[StepLog] = []
        current = initial_view % len(poses)
        visited.add(current)

        grid = VoxelGrid(bounds=cfg.variance.grid_bounds, resolution=cfg.variance.resolution)

        for step in range(cfg.experiment.num_steps):
            rgb, _ = self.sim.render_view(current)
            mask = self.segmenter.segment_affordance(rgb, cfg.segmentation.prompt)

            clouds = self.pointe.generate_point_clouds_from_image(
                rgb, prompt=cfg.pointe.prompt, num_seeds=cfg.pointe.num_seeds, seed_list=cfg.pointe.seed_list
            )

            cam_to_world = pose_to_matrix(poses[current])
            point_weights = [
                compute_point_mask_weights(pc, cam_to_world, cfg.simulator.intrinsics, mask) for pc in clouds
            ]

            variance_grid = compute_variance_field(clouds, cfg.variance)
            semantic_grid = accumulate_semantic_weights(clouds, point_weights, cfg.variance)
            score_grid = combine_variance_and_semantics(variance_grid, semantic_grid)
            centroid, _ = extract_topk_centroid(score_grid, grid, cfg.variance.topk_ratio)

            if cfg.experiment.policy == "active":
                next_view, policy_score = select_next_view_active(current, poses, centroid, visited, cfg.policy)
            elif cfg.experiment.policy == "random":
                next_view = select_next_view_random(current, poses, visited, self.rng)
                policy_score = 0.0
            else:
                next_view = select_next_view_geometric(current, poses, visited)
                policy_score = 0.0

            variance_sum = float(variance_grid.sum())
            logs.append(
                StepLog(
                    step=step,
                    view_id=current,
                    next_view_id=next_view,
                    variance_sum=variance_sum,
                    policy_score=policy_score,
                    centroid=centroid.tolist(),
                )
            )

            self._save_debug(step, rgb, mask, score_grid)
            visited.add(next_view)
            current = next_view

        result = {
            "config": cfg.to_dict(),
            "steps": [asdict(l) for l in logs],
        }
        self._save_logs(result)
        return result

    def _save_logs(self, data: Dict) -> None:
        path = self._output_root / "trajectory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved trajectory log to %s", path)
