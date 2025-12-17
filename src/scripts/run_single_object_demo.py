"""Run a single Active-Hallucination episode for one mesh."""

from __future__ import annotations

import argparse
import sys
import os
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_MESH_DIR = PROJECT_ROOT / "assets" / "meshes"

from src.config import ActiveHallucinationConfig
from src.experiments import ActiveHallucinationRunner


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Single-object Active-Hallucination demo.")
    parser.add_argument("--mesh", type=str, required=True, help="Filename in assets or full path.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config.")
    parser.add_argument("--policy", type=str, choices=["active", "random", "geometric"], default=None, help="Policy.")
    parser.add_argument("--steps", type=int, default=None, help="NBV steps.")
    parser.add_argument("--initial_view", type=int, default=0, help="Start view index.")
    parser.add_argument("--output", type=str, default="outputs/single_demo", help="Output directory.")
    parser.add_argument("--no_vis", action="store_true", help="Skip inline matplotlib visualization.")
    return parser.parse_args()


def resolve_mesh_path(name_or_path: str) -> str:
    path = Path(name_or_path)
    if path.exists(): return str(path)
    asset_path = DEFAULT_MESH_DIR / name_or_path
    if asset_path.exists(): return str(asset_path)
    raise FileNotFoundError(f"Mesh not found: {name_or_path}")


def main() -> None:
    args = parse_args()

    try:
        mesh_path = resolve_mesh_path(args.mesh)
    except FileNotFoundError as e:
        print(e)
        return

    # Setup Config
    cfg = ActiveHallucinationConfig.from_yaml(args.config) if args.config else ActiveHallucinationConfig()
    cfg.simulator.mesh_path = mesh_path
    cfg.experiment.output_dir = args.output
    cfg.experiment.trajectory_name = Path(mesh_path).stem
    cfg.experiment.save_debug = True

    if args.policy: cfg.experiment.policy = args.policy
    if args.steps: cfg.experiment.num_steps = args.steps

    print(f"--- Running Episode: {cfg.experiment.trajectory_name} ---")
    print(f"Policy: {cfg.experiment.policy} | Steps: {cfg.experiment.num_steps}")

    try:
        runner = ActiveHallucinationRunner(cfg)
        result = runner.run_episode(initial_view=args.initial_view)
    except Exception as e:
        print(f"\n[Execution Error] {e}")
        if "pyglet" in str(e) or "EGL" in str(e):
            print("Ensure 'libegl1' is installed and your env supports EGL.")
        return

    out_path = Path(cfg.experiment.output_dir) / cfg.experiment.trajectory_name
    print(f"\nEpisode complete.")
    print(f"Logs and Images saved to: {out_path}")

    # Display final trajectory summary if available
    traj_img = out_path / "fig_trajectory.png"
    if traj_img.exists() and not args.no_vis:
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