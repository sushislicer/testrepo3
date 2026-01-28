"""Lightweight visualization utilities for figures and debugging."""

from __future__ import annotations

import colorsys
import logging
import os
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

logger = logging.getLogger(__name__)


def _is_headless() -> bool:
    return not (os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _render_point_cloud_offscreen_safe(
    point_clouds: List[np.ndarray],
    width: int,
    height: int,
    view_point: Optional[dict],
    timeout_s: float = 8.0,
) -> Tuple[Optional[np.ndarray], Optional[str]]:
    # Run Open3D offscreen rendering in a separate interpreter so EGL/GL failures
    # (including segfaults) don't take down the main process.
    try:
        with tempfile.TemporaryDirectory(prefix="o3d_offscreen_", dir=str(Path.cwd())) as tmpdir:
            tmpdir_path = Path(tmpdir)
            pcs_path = tmpdir_path / "point_clouds.npz"
            out_path = tmpdir_path / "image.npy"

            np.savez_compressed(pcs_path, **{f"pc_{i:03d}": pc for i, pc in enumerate(point_clouds)})

            worker_code = r"""
import colorsys
import os
from pathlib import Path
import numpy as np

pcs_path = Path(os.environ["O3D_PCS_PATH"])
out_path = Path(os.environ["O3D_OUT_PATH"])
width = int(os.environ["O3D_WIDTH"])
height = int(os.environ["O3D_HEIGHT"])

runtime_dir = os.environ.get("XDG_RUNTIME_DIR") or str(out_path.parent / "xdg_runtime")
os.environ["XDG_RUNTIME_DIR"] = runtime_dir
os.makedirs(runtime_dir, exist_ok=True)

import open3d as o3d
from open3d.visualization import rendering

data = np.load(pcs_path)
pcs = [data[k] for k in sorted(data.files)]

renderer = rendering.OffscreenRenderer(width, height)
scene = renderer.scene
scene.set_background([1.0, 1.0, 1.0, 1.0])

has_geo = False
all_points = []
for idx, pc in enumerate(pcs):
    if len(pc) == 0:
        continue
    has_geo = True
    hue = idx / max(len(pcs), 1)
    color = np.array(colorsys.hsv_to_rgb(hue, 0.8, 1.0), dtype=np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc)
    pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (pc.shape[0], 1)))

    mat = rendering.MaterialRecord()
    mat.shader = "defaultUnlit"
    mat.point_size = 3.0
    scene.add_geometry(f"pcd_{idx}", pcd, mat)
    all_points.append(pc)

if not has_geo:
    np.save(out_path, np.zeros((height, width, 3), dtype=np.uint8))
    raise SystemExit(0)

all_points_np = np.concatenate(all_points, axis=0)
center = np.mean(all_points_np, axis=0).astype(np.float32)
extent = (np.max(all_points_np, axis=0) - np.min(all_points_np, axis=0)).astype(np.float32)
radius = float(np.linalg.norm(extent) + 1e-6)

eye = (center + np.array([0.0, 0.0, 2.5 * radius], dtype=np.float32)).astype(np.float32)
up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
renderer.setup_camera(60.0, center, eye, up)

image_o3d = renderer.render_to_image()
image_np = np.asarray(image_o3d)
if hasattr(renderer, "release"):
    renderer.release()
else:
    del renderer

if image_np.ndim == 3 and image_np.shape[-1] == 4:
    image_np = image_np[..., :3]
if image_np.dtype != np.uint8:
    image_np = (np.clip(image_np, 0.0, 1.0) * 255.0).astype(np.uint8)

np.save(out_path, image_np)
"""

            env = os.environ.copy()
            # Prefer headless EGL on NVIDIA in containerized environments:
            # - EGL_PLATFORM=surfaceless avoids requiring X11/Wayland.
            # - Pin vendor to NVIDIA to avoid GLVND selecting Mesa inside Docker.
            if "EGL_PLATFORM" not in env:
                env["EGL_PLATFORM"] = "surfaceless"
            nvidia_vendor_json = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
            if "__EGL_VENDOR_LIBRARY_FILENAMES" not in env and Path(nvidia_vendor_json).exists():
                env["__EGL_VENDOR_LIBRARY_FILENAMES"] = nvidia_vendor_json
            env.update(
                {
                    "O3D_PCS_PATH": str(pcs_path),
                    "O3D_OUT_PATH": str(out_path),
                    "O3D_WIDTH": str(width),
                    "O3D_HEIGHT": str(height),
                }
            )

            try:
                completed = subprocess.run(
                    [sys.executable, "-c", worker_code],
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=timeout_s,
                )
            except subprocess.TimeoutExpired:
                return None, f"OffscreenRenderer timed out after {timeout_s:.1f}s"

            if completed.returncode != 0:
                stderr_tail = (completed.stderr or "").strip().splitlines()[-1:] or [""]
                return None, f"OffscreenRenderer failed (exit={completed.returncode}): {stderr_tail[0]}"

            if not out_path.exists():
                return None, "OffscreenRenderer produced no output"

            try:
                return np.load(out_path), None
            except Exception as e:
                return None, f"OffscreenRenderer output unreadable: {type(e).__name__}: {e}"
    except Exception as e:
        return None, f"OffscreenRenderer wrapper failed: {type(e).__name__}: {e}"


def render_fallback_2d(point_clouds: List[np.ndarray], save_path: Optional[str] = None) -> np.ndarray:
    """
    Matplotlib fallback: Renders a 2D projection (Top-down X-Y) if Open3D fails.
    """
    fig = plt.figure(figsize=(5, 5), dpi=100)
    ax = fig.add_subplot(111)
    
    # Plot clouds
    for idx, pc in enumerate(point_clouds):
        if len(pc) == 0: continue
        # Subsample for speed
        if len(pc) > 1000:
            pc = pc[np.random.choice(len(pc), 1000, replace=False)]
            
        hue = idx / max(len(point_clouds), 1)
        color = colorsys.hsv_to_rgb(hue, 0.8, 1.0)
        
        # X vs Y projection
        ax.scatter(pc[:, 0], pc[:, 1], s=1, color=color, alpha=0.5)

    ax.axis('equal')
    ax.axis('off')
    plt.tight_layout(pad=0)
    
    # Convert figure to numpy array
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()
    if hasattr(fig.canvas, "buffer_rgba"):
        rgba = np.asarray(fig.canvas.buffer_rgba())
        image = np.asarray(rgba, dtype=np.uint8).reshape((h, w, 4))[..., :3]
    else:
        data = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image = data.reshape((h, w, 3))
    
    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path)
        
    plt.close(fig)
    return image


def render_point_cloud_overlay(
    point_clouds: List[np.ndarray],
    save_path: Optional[str] = None,
    view_point: Optional[dict] = None
) -> np.ndarray:
    """
    Render K point clouds. Tries Open3D (3D render) -> Falls back to Matplotlib (2D).
    """
    if o3d is None:
        return render_fallback_2d(point_clouds, save_path)

    width = 800
    height = 800

    # Headless: try OffscreenRenderer (EGL/OSMesa) in a subprocess to avoid hard crashes.
    if _is_headless():
        image_np, err = _render_point_cloud_offscreen_safe(
            point_clouds=point_clouds, width=width, height=height, view_point=view_point
        )
        if image_np is not None:
            if save_path:
                Path(save_path).parent.mkdir(parents=True, exist_ok=True)
                plt.imsave(save_path, image_np)
            return image_np
        if err:
            logger.warning(
                "Open3D OffscreenRenderer unavailable (%s). "
                "Headless Open3D usually needs EGL (GPU) or OSMesa (CPU).",
                err,
            )

    # Attempt legacy Visualizer (often works via X11 / Xvfb, or CPU OSMesa if installed).
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=width, height=height)
        
        has_geo = False
        for idx, pc in enumerate(point_clouds):
            if len(pc) == 0: continue
            has_geo = True
            hue = idx / max(len(point_clouds), 1)
            color = np.array(colorsys.hsv_to_rgb(hue, 0.8, 1.0))
            
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(pc)
            pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (pc.shape[0], 1)))
            vis.add_geometry(pcd)

        if not has_geo:
            vis.destroy_window()
            return np.zeros((height, width, 3), dtype=np.uint8)

        ctr = vis.get_view_control()
        if ctr is None:
            raise RuntimeError("ViewControl failed")

        if not view_point:
            ctr.rotate(10.0, 0.0)

        vis.poll_events()
        vis.update_renderer()
        image = vis.capture_screen_float_buffer(do_render=True)
        image_np = (np.asarray(image) * 255).astype(np.uint8)
        vis.destroy_window()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            plt.imsave(save_path, image_np)
            
        return image_np

    except Exception as e:
        hint = ""
        msg = str(e)
        if _is_headless() and ("OSMesa" in msg or "Failed to create window" in msg):
            hint = (
                " (Headless hint: install `libosmesa6` and set `OPEN3D_CPU_RENDERING=1`, "
                "or run under `xvfb-run`.)"
            )
        logger.warning(f"Open3D rendering failed ({e}). Using 2D fallback.{hint}")
        if 'vis' in locals(): vis.destroy_window()
        return render_fallback_2d(point_clouds, save_path)


def create_ghosting_figure(
    input_rgb: np.ndarray,
    point_clouds: List[np.ndarray],
    variance_grid: np.ndarray,
    save_path: str
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(input_rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")

    pc_render = render_point_cloud_overlay(point_clouds)
    axes[1].imshow(pc_render)
    axes[1].set_title("Multi-seed Point-E")
    axes[1].axis("off")

    proj = np.max(variance_grid, axis=2) if variance_grid.ndim == 3 else variance_grid
    im = axes[2].imshow(proj, cmap="jet", vmin=0, vmax=np.max(proj) + 1e-6)
    axes[2].set_title("Variance Field")
    axes[2].axis("off")
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()


def create_trajectory_figure(steps_data: List[dict], save_path: str) -> None:
    num_steps = len(steps_data)
    fig, axes = plt.subplots(2, num_steps, figsize=(4 * num_steps, 6))
    if num_steps == 1:
        axes = np.expand_dims(axes, axis=1)

    for i, step in enumerate(steps_data):
        ax_rgb = axes[0, i]
        ax_rgb.imshow(step['rgb'])
        ax_rgb.set_title(f"Step {step['step_idx']}")
        ax_rgb.axis("off")
        
        if i < num_steps - 1:
            ax_rgb.arrow(
                x=step['rgb'].shape[1] * 0.9, y=step['rgb'].shape[0] / 2, 
                dx=20, dy=0, head_width=20, color='red'
            )

        ax_var = axes[1, i]
        v_grid = step['variance']
        proj = np.max(v_grid, axis=2) if v_grid.ndim == 3 else v_grid
        ax_var.imshow(proj, cmap="inferno")
        ax_var.set_title(f"Uncertainty: {np.sum(v_grid):.1f}")
        ax_var.axis("off")

    plt.tight_layout()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_vrr_curves(results: Dict[str, List[float]], save_path: str) -> None:
    """Plot VRR (variance reduction rate) curves.

    IMPORTANT: `results[method]` is expected to be a per-step VRR series where:

        vrr[t] = (v0 - v[t]) / max(v0, eps)

    so higher is better and the curve should (ideally) trend upward.

    Historical note: an older version of this function mistakenly plotted
    *normalized variance* (v[t] / v0) while still calling it VRR, which
    inverted the directionality and made debugging confusing.
    """
    plt.figure(figsize=(8, 6))
    for method, vrr_series in results.items():
        steps = range(len(vrr_series))
        plt.plot(steps, vrr_series, marker="o", label=method, linewidth=2)

    plt.title("Variance Reduction Rate (higher is better)")
    plt.xlabel("Step")
    plt.ylabel("VRR")
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.legend()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def plot_success_rates(success_rates: Dict[str, float], save_path: str) -> None:
    plt.figure(figsize=(6, 6))
    methods = list(success_rates.keys())
    rates = list(success_rates.values())
    colors = ['gray', 'orange', 'green']
    bar_colors = [colors[i % len(colors)] for i in range(len(methods))]
    
    plt.bar(methods, rates, color=bar_colors, alpha=0.8)
    plt.ylim(0, 1.05)
    plt.title("Grasp Success Rate")
    plt.ylabel("Success Rate")
    
    for i, v in enumerate(rates):
        plt.text(i, v + 0.02, f"{v:.2f}", ha='center', fontweight='bold')

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def overlay_point_clouds_open3d(point_clouds: List[np.ndarray]) -> None:
    if o3d is None: return
    geoms = []
    for idx, pc in enumerate(point_clouds):
        if len(pc) == 0: continue
        hue = idx / max(len(point_clouds), 1)
        color = np.array(colorsys.hsv_to_rgb(hue, 0.7, 1.0))
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pc)
        pcd.colors = o3d.utility.Vector3dVector(np.tile(color, (pc.shape[0], 1)))
        geoms.append(pcd)
    o3d.visualization.draw_geometries(geoms)


def save_variance_max_projection(score_grid: np.ndarray, path: str) -> None:
    proj = np.max(score_grid, axis=2) if score_grid.ndim == 3 else score_grid
    proj = proj / (proj.max() + 1e-8)
    plt.figure(figsize=(4, 4))
    plt.imshow(proj, cmap="inferno")
    plt.axis("off")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, bbox_inches="tight", pad_inches=0)
    plt.close()
