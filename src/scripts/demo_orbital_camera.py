"""Render an orbital ring of views for a single mesh."""

from __future__ import annotations

import os
# Headless rendering on remote VMs/servers:
#
# IMPORTANT:
# We *force* these variables (not setdefault) because users may have
# `PYOPENGL_PLATFORM=glx` or similar in their shell environment. In that case,
# `setdefault` would keep the incompatible value and pyglet may fail with:
#   "Cannot connect to 'None'".
#
# Set these BEFORE importing pyrender / OpenGL (which happens inside
# [`VirtualTabletopSimulator`](src/simulator.py:96)).
os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["EGL_PLATFORM"] = "surfaceless"
# In some environments pyglet still tries to open an X display; enabling headless
# mode avoids that (supported by pyglet 1.5+).
os.environ["PYGLET_HEADLESS"] = "1"

import argparse
import sys
import math
from pathlib import Path
import imageio
import numpy as np

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

DEFAULT_MESH_DIR = PROJECT_ROOT / "assets" / "meshes"

from src.config import ActiveHallucinationConfig
from src.simulator import VirtualTabletopSimulator


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render orbital camera views.")
    parser.add_argument("--mesh", type=str, required=True, help="Filename in assets or path to mesh.")
    parser.add_argument("--output", type=str, default="outputs/demo_orbit", help="Output directory.")
    parser.add_argument("--num_views", type=int, default=None, help="Override number of views.")
    return parser.parse_args()


def resolve_mesh_path(name_or_path: str) -> str:
    path = Path(name_or_path)
    if path.exists():
        return str(path)
    asset_path = DEFAULT_MESH_DIR / name_or_path
    if asset_path.exists():
        return str(asset_path)
    raise FileNotFoundError(f"Mesh not found: {name_or_path}")


def save_contact_sheet(images: list[np.ndarray], save_path: Path) -> None:
    """Combines all rendered views into a single grid image."""
    if not images: return
    n = len(images)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    
    h, w, c = images[0].shape
    grid = np.zeros((rows * h, cols * w, c), dtype=np.uint8)
    
    for idx, img in enumerate(images):
        r, c_idx = divmod(idx, cols)
        grid[r*h : (r+1)*h, c_idx*w : (c_idx+1)*w] = img
        
    imageio.imwrite(save_path, grid)
    print(f"Saved contact sheet diagram to: {save_path}")


def main() -> None:
    args = parse_args()
    
    try:
        mesh_path = resolve_mesh_path(args.mesh)
        print(f"Loading mesh: {mesh_path}")
    except FileNotFoundError as e:
        print(e)
        return

    cfg = ActiveHallucinationConfig()
    cfg.simulator.mesh_path = mesh_path
    if args.num_views:
        cfg.simulator.num_views = args.num_views

    try:
        sim = VirtualTabletopSimulator(cfg.simulator)
    except Exception as e:
        print(f"\n[Simulator Error] Failed to initialize renderer: {e}")
        print("On Colab, ensure 'libegl1' is installed and EGL headers are present.")
        return

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    rendered_frames = []
    print(f"Rendering {sim.cfg.num_views} views...")
    
    try:
        for i in range(sim.cfg.num_views):
            rgb, _ = sim.render_view(i)
            rgb_uint8 = rgb.astype("uint8")
            rendered_frames.append(rgb_uint8)
            imageio.imwrite(out_dir / f"view_{i:02d}.png", rgb_uint8)
    except Exception as e:
        print(f"Error during rendering loop: {e}")
        return

    # Create summary diagram
    if rendered_frames:
        save_contact_sheet(rendered_frames, out_dir / "all_views_grid.png")
    
    print(f"Saved outputs to {out_dir}")


if __name__ == "__main__":
    main()
