"""Batch runner for multiple meshes and policies."""

from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path

from active_hallucination.config import ActiveHallucinationConfig
from active_hallucination.experiments import ActiveHallucinationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config.")
    parser.add_argument("--mesh_glob", type=str, default="assets/meshes/*.obj", help="Glob pattern for meshes.")
    parser.add_argument("--policies", type=str, default="active,random,geometric", help="Comma list of policies.")
    parser.add_argument("--trials", type=int, default=1, help="Number of trials per mesh/policy.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base_cfg = ActiveHallucinationConfig.from_yaml(args.config) if args.config else ActiveHallucinationConfig()
    meshes = sorted(glob(args.mesh_glob))
    policies = [p.strip() for p in args.policies.split(",") if p.strip()]
    if not meshes:
        print(f"No meshes found for pattern: {args.mesh_glob}")
        return

    for mesh_path in meshes:
        for policy in policies:
            for trial in range(args.trials):
                cfg = ActiveHallucinationConfig.from_dict(base_cfg.to_dict())
                cfg.simulator.mesh_path = mesh_path
                cfg.experiment.policy = policy
                cfg.policy.random_seed = base_cfg.policy.random_seed + trial
                cfg.experiment.trajectory_name = f"{Path(mesh_path).stem}_{policy}_t{trial}"
                runner = ActiveHallucinationRunner(cfg)
                runner.run_episode()
                print(f"Finished {mesh_path} policy={policy} trial={trial}")


if __name__ == "__main__":
    main()
