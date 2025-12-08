"""Render an orbital ring of views for a single mesh."""

from __future__ import annotations

import argparse
from pathlib import Path
import imageio

from active_hallucination.config import ActiveHallucinationConfig
from active_hallucination.simulator import VirtualTabletopSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render orbital camera views for debugging.")
    parser.add_argument("--mesh", type=str, required=True, help="Path to mesh file (OBJ/PLY/GLB/STL).")
    parser.add_argument("--output", type=str, default="outputs/demo_orbit", help="Directory to save renders.")
    parser.add_argument("--num_views", type=int, default=None, help="Override number of orbital views.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = ActiveHallucinationConfig()
    cfg.simulator.mesh_path = args.mesh
    if args.num_views:
        cfg.simulator.num_views = args.num_views
    sim = VirtualTabletopSimulator(cfg.simulator)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    for i in range(sim.cfg.num_views):
        rgb, _ = sim.render_view(i)
        imageio.imwrite(out_dir / f"view_{i:02d}.png", rgb.astype("uint8"))
    print(f"Saved {sim.cfg.num_views} renders to {out_dir}")


if __name__ == "__main__":
    main()
