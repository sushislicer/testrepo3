"""Lightweight visualization utilities for figures and debugging."""

from __future__ import annotations

import colorsys
import logging
from pathlib import Path
from typing import List, Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

try:
    import open3d as o3d
except ImportError:
    o3d = None

logger = logging.getLogger(__name__)


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

    # Attempt Open3D Headless
    try:
        vis = o3d.visualization.Visualizer()
        vis.create_window(visible=False, width=800, height=800)
        
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
            return np.zeros((800, 800, 3), dtype=np.uint8)

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
        logger.warning(f"Open3D rendering failed ({e}). Using 2D fallback.")
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
    plt.figure(figsize=(8, 6))
    for method, variances in results.items():
        steps = range(len(variances))
        initial = variances[0] + 1e-8
        vrr = [v / initial for v in variances]
        plt.plot(steps, vrr, marker='o', label=method, linewidth=2)

    plt.title("Variance Reduction Rate")
    plt.xlabel("Step")
    plt.ylabel("Normalized Variance")
    plt.grid(True, linestyle='--', alpha=0.6)
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
