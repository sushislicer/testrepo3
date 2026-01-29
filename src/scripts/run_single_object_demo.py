"""Run one mesh with run_experiments-like options.

This script is intentionally a "single-mesh batch runner": it supports multiple
policies, multiple trials, and the same initial-view selection modes as
[`run_experiments`](src/scripts/run_experiments.py:49).
"""

from __future__ import annotations

import os
# Prefer EGL headless rendering on servers/VMs.
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("EGL_PLATFORM", "surfaceless")

import argparse
import sys
import zlib
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_MESH_DIR = PROJECT_ROOT / "assets" / "meshes"

from src.config import ActiveHallucinationConfig
from src.experiments import ActiveHallucinationRunner
from src.pointe_wrapper import PointEGenerator
from src.segmentation import CLIPSegSegmenter
from src.simulator import VirtualTabletopSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-object Active-Hallucination demo.")
    parser.add_argument("--mesh", type=str, required=True, help="Filename in assets or full path.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")

    # run_experiments-like batching.
    parser.add_argument(
        "--policies",
        nargs="+",
        default=None,
        help=(
            "Policies to run. Accepts space-separated and/or comma-separated values. "
            "Examples: --policies active active_combined OR --policies active,random"
        ),
    )
    # Backward compatible single-policy flag.
    parser.add_argument(
        "--policy",
        type=str,
        choices=["active", "active_combined", "random", "geometric"],
        default=None,
        help="(Deprecated) Single policy. Prefer --policies.",
    )
    parser.add_argument("--trials", type=int, default=1, help="Trials per policy.")
    parser.add_argument("--steps", type=int, default=None, help="NBV steps per trial.")

    # Initial view selection (same semantics as run_experiments).
    parser.add_argument(
        "--initial_view_mode",
        type=str,
        default="fixed",
        choices=["fixed", "random_per_trial", "sweep", "semantic_occluded"],
        help="How to choose the initial view for each trial.",
    )
    parser.add_argument("--initial_view", type=int, default=0, help="Initial view when mode=fixed.")
    parser.add_argument(
        "--initial_view_seed",
        type=int,
        default=0,
        help="Extra seed offset for initial view selection.",
    )
    parser.add_argument(
        "--initial_view_occlusion_prompt",
        type=str,
        default=None,
        help="Prompt used for semantic_occluded mode (defaults to cfg.segmentation.prompt).",
    )
    parser.add_argument(
        "--initial_view_occlusion_quantile",
        type=float,
        default=0.25,
        help="In semantic_occluded mode, sample from the bottom-q most-occluded views.",
    )

    # Combined-score hyperparameters.
    parser.add_argument(
        "--combined_semantic_variance_weight",
        type=float,
        default=0.5,
        help="semantic_variance_weight used for *_combined policies.",
    )
    parser.add_argument("--semantic_threshold", type=float, default=None)
    parser.add_argument("--semantic_gamma", type=float, default=None)
    parser.add_argument("--semantic_s2_threshold", type=float, default=None)
    parser.add_argument("--semantic_s2_gamma", type=float, default=None)
    parser.add_argument("--semantic_s2_power", type=float, default=None)

    # Output layout.
    parser.add_argument("--output_dir", type=str, default="outputs/single_demo", help="Output root.")
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help="Optional run subfolder name created under --output_dir.",
    )
    parser.add_argument(
        "--no_run_subdir",
        action="store_true",
        help="Write directly into --output_dir (legacy behavior).",
    )

    parser.add_argument("--no_vis", action="store_true", help="Skip inline matplotlib visualization.")
    return parser.parse_args()


def resolve_mesh_path(name_or_path: str) -> str:
    path = Path(name_or_path)
    if path.exists(): return str(path)
    asset_path = DEFAULT_MESH_DIR / name_or_path
    if asset_path.exists(): return str(asset_path)
    raise FileNotFoundError(f"Mesh not found: {name_or_path}")


def _stable_hash32(text: str) -> int:
    """Deterministic (process-stable) 32-bit hash for seeding."""
    return int(zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF)


def _parse_policies(tokens: list[str] | None, fallback: str | None) -> list[str]:
    if tokens:
        out: list[str] = []
        for tok in tokens:
            for p in str(tok).split(","):
                p = p.strip()
                if p:
                    out.append(p)
        # de-dup while preserving order
        seen = set()
        uniq: list[str] = []
        for p in out:
            if p in seen:
                continue
            seen.add(p)
            uniq.append(p)
        return uniq
    if fallback:
        return [str(fallback).strip()]
    return ["active"]


def _choose_initial_view(
    *,
    mode: str,
    fixed_view: int,
    trial: int,
    trials_total: int,
    num_views: int,
    seed: int,
    view_pool: list[int] | None = None,
) -> int:
    num_views = max(int(num_views), 1)
    pool = [int(v) % num_views for v in (view_pool or list(range(num_views)))]
    # De-dup while preserving order.
    seen = set()
    pool = [v for v in pool if not (v in seen or seen.add(v))]
    if not pool:
        pool = list(range(num_views))

    if mode == "fixed":
        return int(fixed_view) % num_views
    if mode == "sweep":
        t = int(trial)
        T = int(trials_total)
        if T <= 1:
            return int(pool[0])
        if T >= len(pool):
            return int(pool[t % len(pool)])
        idx = int(np.floor(t * (len(pool) / float(T))))
        idx = int(np.clip(idx, 0, len(pool) - 1))
        return int(pool[idx])

    rng = np.random.RandomState(int(seed) + int(trial) * 10_007)
    return int(rng.choice(pool))


def _compute_semantic_occlusion_pool(
    *,
    mesh_path: str,
    base_cfg: ActiveHallucinationConfig,
    segmenter: CLIPSegSegmenter,
    prompt: str,
    quantile: float,
) -> list[int]:
    q = float(np.clip(float(quantile), 0.0, 1.0))
    cfg = ActiveHallucinationConfig.from_dict(base_cfg.to_dict())
    cfg.simulator.mesh_path = mesh_path
    sim = VirtualTabletopSimulator(cfg.simulator)
    try:
        scores: list[float] = []
        thr = float(getattr(cfg.segmentation, "threshold", 0.25))
        for view_id in range(int(cfg.simulator.num_views)):
            rgb, _ = sim.render_view(view_id)
            mask = segmenter.segment_affordance(rgb, prompt)
            scores.append(float(np.mean(mask > thr)))
        if not scores:
            return []
        cutoff = float(np.quantile(np.asarray(scores, dtype=np.float32), q))
        return [i for i, s in enumerate(scores) if float(s) <= cutoff + 1e-12]
    finally:
        try:
            sim.close()
        except Exception:
            pass


def main() -> None:
    args = parse_args()

    try:
        mesh_path = resolve_mesh_path(args.mesh)
    except FileNotFoundError as e:
        print(e)
        return

    base_cfg = ActiveHallucinationConfig.from_yaml(args.config) if args.config else ActiveHallucinationConfig()
    base_cfg.simulator.mesh_path = mesh_path
    base_cfg.experiment.save_debug = True
    if args.steps:
        base_cfg.experiment.num_steps = int(args.steps)

    policies = _parse_policies(args.policies, args.policy)
    trials = int(max(1, args.trials))

    run_name = str(args.run_name) if args.run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_root = Path(args.output_dir)
    if not args.no_run_subdir:
        output_root = output_root / run_name
    mesh_name = Path(mesh_path).stem
    mesh_root = output_root / mesh_name
    (mesh_root / "figures").mkdir(parents=True, exist_ok=True)

    # Shared heavy models once.
    shared_pointe = PointEGenerator(base_cfg.pointe)
    shared_segmenter = CLIPSegSegmenter(base_cfg.segmentation)

    # Optional semantic-occlusion pool (computed once).
    view_pool: list[int] | None = None
    if str(args.initial_view_mode) == "semantic_occluded":
        occl_prompt = args.initial_view_occlusion_prompt or base_cfg.segmentation.prompt
        view_pool = _compute_semantic_occlusion_pool(
            mesh_path=str(mesh_path),
            base_cfg=base_cfg,
            segmenter=shared_segmenter,
            prompt=str(occl_prompt),
            quantile=float(args.initial_view_occlusion_quantile),
        )
        if view_pool:
            print(
                f"[InitView] semantic_occluded pool size={len(view_pool)}/{int(base_cfg.simulator.num_views)} prompt='{occl_prompt}'",
                flush=True,
            )

    mesh_seed = int(base_cfg.policy.random_seed) + int(args.initial_view_seed) + _stable_hash32(mesh_name)

    print(f"Output root: {mesh_root}")
    print(f"Policies: {policies} | Trials: {trials} | Steps: {int(base_cfg.experiment.num_steps)}")

    last_traj_dir: Path | None = None
    for policy in policies:
        for trial in range(trials):
            initial_view = _choose_initial_view(
                mode=str(args.initial_view_mode),
                fixed_view=int(args.initial_view),
                trial=int(trial),
                trials_total=int(trials),
                num_views=int(base_cfg.simulator.num_views),
                seed=int(mesh_seed),
                view_pool=view_pool,
            )

            cfg = ActiveHallucinationConfig.from_dict(base_cfg.to_dict())
            cfg.simulator.mesh_path = mesh_path

            # Optional hyperparameter overrides.
            if args.semantic_threshold is not None:
                cfg.variance.semantic_threshold = float(args.semantic_threshold)
            if args.semantic_gamma is not None:
                cfg.variance.semantic_gamma = float(args.semantic_gamma)
            if args.semantic_s2_threshold is not None:
                cfg.variance.semantic_s2_threshold = float(args.semantic_s2_threshold)
            if args.semantic_s2_gamma is not None:
                cfg.variance.semantic_s2_gamma = float(args.semantic_s2_gamma)
            if args.semantic_s2_power is not None:
                cfg.variance.semantic_s2_power = float(args.semantic_s2_power)

            # Deterministic trial seeds.
            cfg.policy.random_seed = int(base_cfg.policy.random_seed) + int(trial)
            seed_base = int(base_cfg.policy.random_seed) + int(trial) * 1000
            cfg.pointe.seed_list = [seed_base + i for i in range(int(cfg.pointe.num_seeds))]

            # Combined policy variant.
            actual_policy = policy
            if policy.endswith("_combined"):
                actual_policy = policy.replace("_combined", "")
                cfg.variance.semantic_variance_weight = float(args.combined_semantic_variance_weight)
            cfg.experiment.policy = actual_policy

            cfg.experiment.output_dir = str(mesh_root)
            cfg.experiment.trajectory_name = f"{policy}_t{trial}"

            print(
                f"[Run] {mesh_name} | {policy} | T{trial} | init_view={initial_view}",
                flush=True,
            )
            try:
                runner = ActiveHallucinationRunner(cfg, sim=None, pointe=shared_pointe, segmenter=shared_segmenter)
                runner.run_episode(initial_view=int(initial_view))
            except Exception as e:
                print(f"[Run] Failed {mesh_name} {policy} T{trial}: {e}", flush=True)
                if "pyglet" in str(e) or "EGL" in str(e):
                    print("Ensure 'libegl1' is installed and your env supports EGL.")
            finally:
                try:
                    runner.close()
                except Exception:
                    pass

            last_traj_dir = mesh_root / f"{policy}_t{trial}"

    print("\nAll trials complete.")
    print(f"Logs and Images saved under: {mesh_root}")

    # Display final trajectory summary if available
    traj_img = (last_traj_dir / "fig_trajectory.png") if last_traj_dir else None
    if traj_img is not None and traj_img.exists() and not args.no_vis:
        try:
            img = mpimg.imread(str(traj_img))
            plt.figure(figsize=(15, 5))
            plt.imshow(img)
            plt.axis('off')
            plt.title("Trajectory Summary")
            plt.show()
        except Exception:
            print(f"Could not display inline image. View file at: {traj_img}")


if __name__ == "__main__":
    main()
