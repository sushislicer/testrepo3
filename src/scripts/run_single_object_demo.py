"""Run a single Active-Hallucination episode for one mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.config import ActiveHallucinationConfig
from src.experiments import ActiveHallucinationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-object Active-Hallucination demo.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to mesh file.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config to override defaults.")
    parser.add_argument("--policy", type=str, choices=["active", "random", "geometric"], default=None, help="Policy to use.")
    parser.add_argument("--steps", type=int, default=None, help="Number of NBV steps.")
    parser.add_argument("--initial_view", type=int, default=0, help="Index of initial camera view.")
    parser.add_argument("--output", type=str, default=None, help="Output directory (overrides config).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ActiveHallucinationConfig.from_yaml(args.config) if args.config else ActiveHallucinationConfig()
    cfg.simulator.mesh_path = args.mesh
    if args.policy:
        cfg.experiment.policy = args.policy
    if args.steps:
        cfg.experiment.num_steps = args.steps
    if args.output:
        cfg.experiment.output_dir = args.output
    cfg.experiment.trajectory_name = Path(args.mesh).stem

    runner = ActiveHallucinationRunner(cfg)
    result = runner.run_episode(initial_view=args.initial_view)
    print(f"Episode complete. Logs saved under {cfg.experiment.output_dir}/{cfg.experiment.trajectory_name}")
    print(f"Steps: {len(result['steps'])}")


if __name__ == "__main__":
    main()
