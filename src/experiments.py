"""Experiment orchestration for Active-Hallucination."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np

from .config import ActiveHallucinationConfig
from .cloud_frame import (
    compute_object_mask,
    estimate_orbit_distance_for_cloud,
    estimate_yaw_align_to_mask,
    make_cloud_orbital_poses,
    normalize_clouds_to_bounds,
    rotate_clouds_yaw,
)
from .pointe_wrapper import PointEGenerator
from .segmentation import (
    CLIPSegSegmenter,
    compute_point_mask_weights_multiview,
    render_point_cloud_view,
)
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
from .visualization import (
    create_ghosting_figure,
    create_trajectory_figure,
    plot_vrr_curves,
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
    distance_to_gt: Optional[float] = None
    success: Optional[bool] = None


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

    def _save_debug(
        self,
        step: int,
        rgb: np.ndarray,
        score_grid: np.ndarray,
        *,
        semantic_render: Optional[np.ndarray] = None,
        semantic_mask: Optional[np.ndarray] = None,
    ) -> None:
        if not self.cfg.experiment.save_debug:
            return
        import imageio

        imageio.imwrite(self._output_root / f"step_{step:02d}_rgb.png", rgb.astype(np.uint8))

        if semantic_render is not None:
            imageio.imwrite(
                self._output_root / f"step_{step:02d}_cloud0_view0_render.png", semantic_render.astype(np.uint8)
            )
        if semantic_mask is not None:
            imageio.imwrite(
                self._output_root / f"step_{step:02d}_cloud0_view0_mask.png", (semantic_mask * 255).astype(np.uint8)
            )

        proj = np.max(score_grid, axis=2)
        proj = (proj / (proj.max() + 1e-8) * 255).astype(np.uint8)
        imageio.imwrite(self._output_root / f"step_{step:02d}_score.png", proj)

    def run_episode(self, initial_view: int = 0) -> Dict:
        cfg = self.cfg
        poses_sim = self.sim.list_poses()
        visited: Set[int] = set()
        logs: List[StepLog] = []
        vis_trajectory: List[Dict] = []
        
        current = initial_view % len(poses_sim)
        visited.add(current)

        grid = VoxelGrid(bounds=cfg.variance.grid_bounds, resolution=cfg.variance.resolution)
        gt = (
            np.array(cfg.experiment.gt_handle_centroid, dtype=np.float32)
            if cfg.experiment.gt_handle_centroid is not None
            else None
        )
        # Track a persistent yaw alignment so view-index mapping stays stable even when
        # later views are silhouette-ambiguous (e.g., handle fully occluded).
        global_yaw_deg: Optional[float] = None

        for step in range(cfg.experiment.num_steps):
            rgb, _ = self.sim.render_view(current)

            clouds = self.pointe.generate_point_clouds_from_image(
                rgb, prompt=cfg.pointe.prompt, num_seeds=cfg.pointe.num_seeds, seed_list=cfg.pointe.seed_list
            )

            # --- Cloud-frame normalization + view alignment ---
            # Normalize into an object-centric frame so voxelization/rendering is coherent.
            clouds_norm, _norm = normalize_clouds_to_bounds(
                clouds, grid_bounds=cfg.variance.grid_bounds, margin=0.95, center_method="median"
            )
            # Define an orbital camera in the *same cloud frame* (fixes starfield renders).
            # Use the known view index so the "front face" is meaningful for hallucinations.
            all_pts = np.concatenate([pc for pc in clouds_norm if pc.size > 0], axis=0) if clouds_norm else np.zeros((0, 3), dtype=np.float32)
            cloud_radius = estimate_orbit_distance_for_cloud(
                all_pts if all_pts.size > 0 else (clouds_norm[0] if clouds_norm else np.zeros((0, 3), dtype=np.float32)),
                cfg.simulator.intrinsics,
                target_fill=getattr(cfg.simulator, "target_fill", 0.65),
            )
            poses_cloud = make_cloud_orbital_poses(
                num_views=cfg.simulator.num_views, elevation_deg=cfg.simulator.elevation_deg, radius=cloud_radius
            )
            base_cam_to_world = pose_to_matrix(poses_cloud[current])

            # Estimate a global yaw that aligns the point cloud's silhouette to the
            # conditioning image silhouette for this view. This enforces "front-face
            # alignment" so occluded regions are consistent in 3D.
            input_mask = compute_object_mask(rgb, threshold=5)
            yaw_deg = estimate_yaw_align_to_mask(
                clouds_norm[0] if clouds_norm else np.zeros((0, 3), dtype=np.float32),
                input_mask=input_mask,
                base_cam_to_world=base_cam_to_world,
                intrinsics=cfg.simulator.intrinsics,
                prior_yaw_deg=global_yaw_deg,
                prior_window_deg=30.0,
                coarse_step_deg=10.0 if global_yaw_deg is None else 5.0,
                fine_step_deg=2.0,
                point_radius_px=cfg.segmentation.render_point_radius_px,
                blur_sigma=cfg.segmentation.render_blur_sigma,
                render_threshold=10,
            )
            global_yaw_deg = yaw_deg
            clouds_aligned = rotate_clouds_yaw(clouds_norm, yaw_deg=yaw_deg)

            # Semantic painting happens per-PointE generation: render each cloud from multiple
            # azimuths (0, 120, 240 deg by default), run CLIPSeg per render, then project masks
            # back to 3D points to obtain per-point semantic weights.
            point_weights = []
            semantic_render0 = None
            semantic_mask0 = None
            for cloud_idx, pc in enumerate(clouds_aligned):
                w = compute_point_mask_weights_multiview(
                    pc,
                    base_cam_to_world,
                    cfg.simulator.intrinsics,
                    self.segmenter,
                    prompt=cfg.segmentation.prompt,
                    num_views=cfg.segmentation.views_per_cloud,
                    yaw_step_deg=cfg.segmentation.yaw_step_deg,
                    aggregation=cfg.segmentation.aggregation,
                    point_radius_px=cfg.segmentation.render_point_radius_px,
                    blur_sigma=cfg.segmentation.render_blur_sigma,
                )
                point_weights.append(w)

                # Save one representative semantic render/mask for debugging (cloud 0, view 0).
                if cfg.experiment.save_debug and cloud_idx == 0:
                    semantic_render0 = render_point_cloud_view(
                        pc,
                        base_cam_to_world,
                        cfg.simulator.intrinsics,
                        point_radius_px=cfg.segmentation.render_point_radius_px,
                        blur_sigma=cfg.segmentation.render_blur_sigma,
                    )
                    semantic_mask0 = self.segmenter.segment_affordance(semantic_render0, cfg.segmentation.prompt)

            variance_grid = compute_variance_field(clouds_aligned, cfg.variance)
            semantic_stack = accumulate_semantic_weights(clouds_aligned, point_weights, cfg.variance)
            score_grid = combine_variance_and_semantics(variance_grid, semantic_stack, cfg.variance)
            centroid, _ = extract_topk_centroid(score_grid, grid, cfg.variance.topk_ratio)

            # --- Visualization Hooks ---
            if cfg.experiment.save_debug:
                # Ghosting Figure (Step 0)
                if step == 0:
                    create_ghosting_figure(
                        rgb, clouds_aligned, variance_grid, 
                        str(self._output_root / "fig_ghosting_step0.png")
                    )
                # Collect data for trajectory strip
                vis_trajectory.append({
                    "rgb": rgb,
                    "variance": score_grid,  # visualizing combined score for trajectory
                    "step_idx": step
                })

            distance_to_gt = None
            success_flag = None
            if gt is not None:
                # Map centroid (cloud frame) into the simulator world frame approximately
                # via camera-distance scale matching, then translate to the simulator look_at.
                sim_pose = poses_sim[current]
                d_world = float(np.linalg.norm(sim_pose.position - sim_pose.look_at))
                d_cloud = float(np.linalg.norm(poses_cloud[current].position - poses_cloud[current].look_at))
                scale = d_world / max(d_cloud, 1e-6)
                centroid_world = sim_pose.look_at + centroid * float(scale)
                distance_to_gt = float(np.linalg.norm(centroid_world - gt))
                success_flag = distance_to_gt <= cfg.experiment.success_threshold

            if cfg.experiment.policy == "active":
                # Use cloud-frame poses so centroid and camera live in the same coordinate frame.
                next_view, policy_score = select_next_view_active(current, poses_cloud, centroid, visited, cfg.policy)
            elif cfg.experiment.policy == "random":
                next_view = select_next_view_random(current, poses_cloud, visited, self.rng)
                policy_score = 0.0
            else:
                next_view = select_next_view_geometric(current, poses_cloud, visited)
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
                    distance_to_gt=distance_to_gt,
                    success=success_flag,
                )
            )

            self._save_debug(step, rgb, score_grid, semantic_render=semantic_render0, semantic_mask=semantic_mask0)
            visited.add(next_view)
            current = next_view

        variance_sums = [log.variance_sum for log in logs]
        vrr = self._compute_vrr(variance_sums)
        success_at_k = logs[-1].success if logs else None
        dist_at_k = logs[-1].distance_to_gt if logs else None

        # --- Final Visualization ---
        if cfg.experiment.save_debug:
            create_trajectory_figure(vis_trajectory, str(self._output_root / "fig_trajectory.png"))
            plot_vrr_curves(
                {cfg.experiment.policy: variance_sums}, 
                str(self._output_root / "fig_vrr.png")
            )

        result = {
            "config": cfg.to_dict(),
            "steps": [asdict(l) for l in logs],
            "metrics": {
                "variance_sum": variance_sums,
                "variance_reduction_rate": vrr,
                "success_at_final_step": success_at_k,
                "distance_to_gt_at_final_step": dist_at_k,
            },
        }
        self._save_logs(result)
        self._save_csv(logs, vrr)
        return result

    def _save_logs(self, data: Dict) -> None:
        path = self._output_root / "trajectory.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info("Saved trajectory log to %s", path)

    @staticmethod
    def _compute_vrr(variance_sums: List[float]) -> List[float]:
        """Variance Reduction Rate per step, normalized to step 0."""
        if not variance_sums:
            return []
        baseline = variance_sums[0]
        denom = baseline if baseline > 1e-8 else 1e-8
        return [(baseline - v) / denom for v in variance_sums]

    def _save_csv(self, logs: List[StepLog], vrr: List[float]) -> None:
        path = self._output_root / "trajectory_metrics.csv"
        import csv

        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "step",
                    "view_id",
                    "next_view_id",
                    "variance_sum",
                    "variance_reduction_rate",
                    "policy_score",
                    "centroid_x",
                    "centroid_y",
                    "centroid_z",
                    "distance_to_gt",
                    "success",
                ]
            )
            for log, vrr_val in zip(logs, vrr):
                cx, cy, cz = log.centroid
                writer.writerow(
                    [
                        log.step,
                        log.view_id,
                        log.next_view_id,
                        log.variance_sum,
                        vrr_val,
                        log.policy_score,
                        cx,
                        cy,
                        cz,
                        log.distance_to_gt if log.distance_to_gt is not None else "",
                        log.success if log.success is not None else "",
                    ]
                )
        logger.info("Saved trajectory metrics CSV to %s", path)
