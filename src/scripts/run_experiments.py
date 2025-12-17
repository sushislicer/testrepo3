"""Batch runner for multiple meshes and policies."""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from glob import glob
from pathlib import Path

import numpy as np

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.config import ActiveHallucinationConfig
from src.experiments import ActiveHallucinationRunner
from src.visualization import plot_success_rates, plot_vrr_curves


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config.")
    parser.add_argument("--mesh_glob", type=str, default="assets/meshes/*.obj", help="Glob pattern.")
    parser.add_argument("--policies", type=str, default="active,random,geometric", help="Policies list.")
    parser.add_argument("--trials", type=int, default=1, help="Trials per setting.")
    parser.add_argument("--output_dir", type=str, default="outputs/batch_results", help="Output dir.")
    return parser.parse_args()


def resolve_meshes(glob_pattern: str) -> list[str]:
    files = sorted(glob(glob_pattern))
    if files: return files
    root_pattern = str(PROJECT_ROOT / glob_pattern)
    return sorted(glob(root_pattern))


def main() -> None:
    args = parse_args()
    base_cfg = ActiveHallucinationConfig.from_yaml(args.config) if args.config else ActiveHallucinationConfig()
    
    meshes = resolve_meshes(args.mesh_glob)
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    
    if not meshes:
        print(f"No meshes found: {args.mesh_glob}")
        return

    agg_results = defaultdict(lambda: {"vrr": [], "success": []})
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    for mesh_path in meshes:
        mesh_name = Path(mesh_path).stem
        for policy in policies:
            for trial in range(args.trials):
                cfg = ActiveHallucinationConfig.from_dict(base_cfg.to_dict())
                cfg.simulator.mesh_path = mesh_path
                cfg.experiment.policy = policy
                cfg.policy.random_seed = base_cfg.policy.random_seed + trial
                
                cfg.experiment.output_dir = str(output_root / mesh_name)
                cfg.experiment.trajectory_name = f"{policy}_t{trial}"

                runner = ActiveHallucinationRunner(cfg)
                try:
                    result = runner.run_episode()
                    metrics = result["metrics"]
                    agg_results[policy]["vrr"].append(metrics["variance_reduction_rate"])
                    agg_results[policy]["success"].append(1.0 if metrics["success_at_final_step"] else 0.0)
                    print(f"Finished {mesh_name} | {policy} | T{trial}")
                except Exception as e:
                    print(f"Failed {mesh_name} {policy} T{trial}: {e}")

    avg_vrr_map = {}
    for policy, data in agg_results.items():
        vrrs = data["vrr"]
        if not vrrs: continue
        min_len = min(len(v) for v in vrrs)
        mat = np.array([v[:min_len] for v in vrrs])
        avg_vrr_map[policy] = np.mean(mat, axis=0).tolist()

    plot_vrr_curves(avg_vrr_map, str(output_root / "figures" / "comparison_vrr.png"))

    success_map = {}
    for policy, data in agg_results.items():
        outcomes = data["success"]
        success_map[policy] = float(np.mean(outcomes)) if outcomes else 0.0

    plot_success_rates(success_map, str(output_root / "figures" / "comparison_success.png"))
    print(f"Batch run complete. Outputs in {output_root}")


if __name__ == "__main__":
    main()