"""Batch runner for multiple meshes and policies."""

from __future__ import annotations

import os
# CRITICAL FIX: Set headless rendering backend before importing visualization libs
# This prevents Open3D/GLFW from trying to open a window and timing out.
os.environ["PYOPENGL_PLATFORM"] = "egl"
# Ensure EGL uses surfaceless platform to avoid display dependency and potential conflicts
os.environ["EGL_PLATFORM"] = "surfaceless"

# Monkeypatch collections for Python 3.10+ compatibility (needed by older pyrender/networkx)
import collections
import collections.abc
if not hasattr(collections, "Mapping"):
    collections.Mapping = collections.abc.Mapping
if not hasattr(collections, "MutableMapping"):
    collections.MutableMapping = collections.abc.MutableMapping
if not hasattr(collections, "Set"):
    collections.Set = collections.abc.Set
if not hasattr(collections, "MutableSet"):
    collections.MutableSet = collections.abc.MutableSet
if not hasattr(collections, "Iterable"):
    collections.Iterable = collections.abc.Iterable

import argparse
import sys
import zlib
from datetime import datetime
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
from src.simulator import VirtualTabletopSimulator
from src.pointe_wrapper import PointEGenerator
from src.segmentation import CLIPSegSegmenter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run batch experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config.")
    parser.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["mugs5", "objaverse_subset"],
        help=(
            "Convenience dataset selector. "
            "'mugs5' runs all OBJ meshes under assets/ whose filename contains 'mug' (case-insensitive). "
            "'objaverse_subset' runs the 4 categories (mugs, pans, kettles, hammers) in assets/objaverse_subset."
        ),
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        default="assets",
        help="Project-relative assets root used by presets.",
    )
    parser.add_argument("--mesh_glob", type=str, default="assets/meshes/*.obj", help="Glob pattern.")
    parser.add_argument("--mesh_list", type=str, default=None, help="Path to text file with list of meshes (overrides mesh_glob).")
    parser.add_argument(
        "--objects",
        type=str,
        default=None,
        help=(
            "Object categories to run (comma-separated, e.g. 'mug,pitcher'). "
            "If set to 'all', runs all meshes under --assets_dir. "
            "If provided, this overrides --mesh_glob unless --mesh_list is provided."
        ),
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="assets/meshes",
        help="Root directory containing category subfolders for --objects.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["active", "random", "geometric"],
        help=(
            "Policies to run. Accepts space-separated (recommended) and/or comma-separated values. "
            "Examples: --policies active random geometric active_combined OR --policies active,random,geometric"
        ),
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per setting.")
    parser.add_argument("--output_dir", type=str, default="outputs/batch_results", help="Output dir.")

    # Mesh pose normalization.
    # Some real-world assets (e.g., Objaverse) come in arbitrary coordinate frames,
    # so stable-pose upright normalization can help. However, the 5 shipped mug
    # assets are already oriented correctly and stable-pose can choose a sideways
    # placement (cup lying on its side), which is undesirable for the mugs5 preset.
    parser.add_argument(
        "--normalize_stable_pose",
        type=str,
        default="auto",
        choices=["auto", "on", "off"],
        help=(
            "Control simulator stable-pose normalization. "
            "auto: disable for --preset mugs5, enable otherwise. "
            "on/off: force enable/disable."
        ),
    )

    # Run directory handling.
    parser.add_argument(
        "--run_name",
        type=str,
        default=None,
        help=(
            "Optional run subfolder name created under --output_dir. "
            "If omitted (default), an auto timestamped folder like run_YYYYMMDD_HHMMSS is used."
        ),
    )
    parser.add_argument(
        "--no_run_subdir",
        action="store_true",
        help="Write results directly into --output_dir (legacy behavior).",
    )

    # View-trajectory diversification.
    # The original code always started at view 0, which makes many trials/policies
    # follow nearly identical trajectories on symmetric objects (e.g., mugs) and
    # yields a poor estimate of VRR "from each angle".
    parser.add_argument(
        "--initial_view_mode",
        type=str,
        default="random_per_trial",
        choices=["fixed", "random_per_trial", "sweep", "semantic_occluded"],
        help=(
            "How to choose the initial camera view for each (mesh,policy,trial). "
            "fixed: always use --initial_view. "
            "random_per_trial: deterministically sample a different start view per trial using the base random_seed. "
            "sweep: evenly-space initial views across the orbit (e.g., trials=3 -> 0, 8, 16 when num_views=24). "
            "semantic_occluded: pick an initial view from the lowest-visibility semantic views (requires rendering all orbital views once per mesh)."
        ),
    )
    parser.add_argument(
        "--initial_view",
        type=int,
        default=0,
        help="Initial view index used when --initial_view_mode=fixed.",
    )
    parser.add_argument(
        "--initial_view_seed",
        type=int,
        default=0,
        help=(
            "Extra seed offset for initial view selection. "
            "Useful to re-sample start views across repeated runs while keeping determinism within a run."
        ),
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
        help=(
            "In semantic_occluded mode, sample from views whose semantic visibility is in the bottom q-quantile. "
            "Example: 0.25 means bottom 25%% most-occluded views."
        ),
    )

    # Combined-score hyperparameters (so you can sweep without editing YAML).
    parser.add_argument(
        "--combined_semantic_variance_weight",
        type=float,
        default=0.5,
        help="semantic_variance_weight used for *_combined policies.",
    )
    parser.add_argument(
        "--semantic_threshold",
        type=float,
        default=None,
        help="Override cfg.variance.semantic_threshold (if provided).",
    )
    parser.add_argument(
        "--semantic_gamma",
        type=float,
        default=None,
        help="Override cfg.variance.semantic_gamma (if provided).",
    )

    # Secondary semantic signal (S2) shaping overrides.
    parser.add_argument(
        "--semantic_s2_threshold",
        type=float,
        default=None,
        help="Override cfg.variance.semantic_s2_threshold (if provided).",
    )
    parser.add_argument(
        "--semantic_s2_gamma",
        type=float,
        default=None,
        help="Override cfg.variance.semantic_s2_gamma (if provided).",
    )
    parser.add_argument(
        "--semantic_s2_power",
        type=float,
        default=None,
        help="Override cfg.variance.semantic_s2_power (if provided).",
    )
    return parser.parse_args()


def _stable_hash32(text: str) -> int:
    """Deterministic (process-stable) 32-bit hash for seeding.

    Avoids Python's built-in hash(), which is salted per-process.
    """
    return int(zlib.adler32(text.encode("utf-8")) & 0xFFFFFFFF)


def _is_objaverse_mesh(mesh_path: str) -> bool:
    """Heuristic: treat Objaverse assets as needing stable-pose normalization.

    Curated meshes (e.g., our shipped mug assets) are often already upright.
    Trimesh's stable-pose selection can lay these on their side, which makes
    the rendered mug look sideways even though the orbital camera is correct.
    """
    p = str(mesh_path).replace("\\", "/").lower()
    return ("/objaverse/" in p) or ("/objaverse_subset/" in p)


def _choose_initial_view(
    *,
    mode: str,
    fixed_view: int,
    trial: int,
    trials_total: int | None = None,
    num_views: int,
    seed: int,
    view_pool: list[int] | None = None,
) -> int:
    """Deterministically choose an initial view.

    Determinism is important so that different policies can be compared fairly
    (same start view for the same trial index).
    """
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
        # Evenly-space initial views across the orbit.
        # Example: num_views=24, trials_total=3 -> [0, 8, 16] instead of [0, 1, 2].
        t = int(trial)
        T = int(trials_total) if trials_total is not None else 0
        if T <= 1:
            return int(pool[0])
        # If we have more trials than distinct views, fall back to wrapping.
        if T >= len(pool):
            return int(pool[t % len(pool)])
        idx = int(np.floor(t * (len(pool) / float(T))))
        idx = int(np.clip(idx, 0, len(pool) - 1))
        return int(pool[idx])
    # mode == "random_per_trial" | "semantic_occluded"
    rng = np.random.RandomState(int(seed) + int(trial) * 10_007)
    return int(rng.choice(pool))


def _compute_semantic_occlusion_pool(
    *,
    mesh_path: str,
    base_cfg: ActiveHallucinationConfig,
    segmenter: CLIPSegSegmenter,
    prompt: str,
    quantile: float,
    normalize_stable_pose: bool | None = None,
) -> list[int]:
    """Return a list of view indices where the semantic region is least visible.

    This uses CLIPSeg on the *simulator RGB* (not point clouds). It's a cheap proxy
    for "handle is occluded" and is only used to diversify initial conditions.
    """
    q = float(np.clip(quantile, 0.0, 1.0))
    cfg = ActiveHallucinationConfig.from_dict(base_cfg.to_dict())
    cfg.simulator.mesh_path = mesh_path
    if normalize_stable_pose is not None:
        cfg.simulator.normalize_stable_pose = bool(normalize_stable_pose)
    sim = VirtualTabletopSimulator(cfg.simulator)
    try:
        scores: list[float] = []
        for view_id in range(int(cfg.simulator.num_views)):
            rgb, _ = sim.render_view(view_id)
            mask = segmenter.segment_affordance(rgb, prompt)
            # Visibility score: fraction of pixels predicted as semantic region.
            scores.append(float(np.mean(mask > 0.5)))
        if not scores:
            return []
        thr = float(np.quantile(np.asarray(scores, dtype=np.float32), q))
        pool = [i for i, s in enumerate(scores) if float(s) <= thr + 1e-12]
        return pool
    finally:
        try:
            sim.close()
        except Exception:
            pass


def resolve_meshes(glob_pattern: str) -> list[str]:
    files = sorted(glob(glob_pattern))
    if files: return files
    root_pattern = str(PROJECT_ROOT / glob_pattern)
    return sorted(glob(root_pattern))


def resolve_meshes_from_objects(objects: str, assets_dir: str) -> list[str]:
    """Resolve meshes by category folders under assets_dir.

    Expected layout:
      assets_dir/<category>/**/*.(obj|ply|stl|glb|gltf)
    """
    assets_path = Path(assets_dir)
    if not assets_path.is_absolute():
        assets_path = PROJECT_ROOT / assets_path

    valid_ext = {".obj", ".ply", ".stl", ".glb", ".gltf"}

    obj = (objects or "").strip().lower()
    if obj == "all":
        search_roots = [assets_path]
    else:
        cats = [c.strip() for c in obj.split(",") if c.strip()]
        search_roots = [assets_path / c for c in cats]

    files: list[str] = []
    for root in search_roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in valid_ext:
                files.append(str(p))
    return sorted(set(files))


def resolve_meshes_from_preset(preset: str, assets_root: str) -> list[str]:
    preset = (preset or "").strip().lower()
    root = Path(assets_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    if preset == "mugs5":
        # Strictly run ONLY the 5 OBJ files shipped at assets/*.obj.
        # Do not recurse into assets/objaverse_subset or assets/meshes.
        return sorted(str(p) for p in root.glob("*.obj"))

    if preset == "objaverse_subset":
        # Expected path: assets/objaverse_subset/{category}/*.obj
        subset_root = root / "objaverse_subset"
        if not subset_root.exists():
            if (root / "mugs").exists():
                subset_root = root
        
        categories = ["mugs", "pans", "kettles", "hammers"]
        out: list[str] = []
        for cat in categories:
            cat_dir = subset_root / cat
            if not cat_dir.exists():
                continue
            for p in cat_dir.glob("*.obj"):
                out.append(str(p))
        return sorted(set(out))

    raise ValueError(f"Unknown preset '{preset}'")


def _parse_policies(policies_tokens: list[str]) -> list[str]:
    # Support both comma-separated and space-separated forms.
    out: list[str] = []
    for tok in policies_tokens:
        for p in str(tok).split(","):
            p = p.strip()
            if p:
                out.append(p)
    # Keep order but de-dup.
    seen = set()
    uniq = []
    for p in out:
        if p in seen:
            continue
        seen.add(p)
        uniq.append(p)
    return uniq


def main() -> None:
    args = parse_args()
    base_cfg = ActiveHallucinationConfig.from_yaml(args.config) if args.config else ActiveHallucinationConfig()

    # Stable-pose normalization mode.
    # NOTE: We apply the final decision per-mesh (later), because in "auto" mode
    # curated mug meshes should stay upright, while Objaverse assets often need
    # reorientation.
    normalize_mode = str(args.normalize_stable_pose)
    if normalize_mode == "on":
        base_cfg.simulator.normalize_stable_pose = True
    elif normalize_mode == "off":
        base_cfg.simulator.normalize_stable_pose = False

    # Create a fresh run subdirectory by default to avoid overwriting previous results.
    # This makes it easier to compare different hyperparameter sweeps.
    run_name = str(args.run_name) if args.run_name else datetime.now().strftime("run_%Y%m%d_%H%M%S")
    
    if args.mesh_list:
        with open(args.mesh_list, 'r') as f:
            meshes = [line.strip() for line in f if line.strip()]
    elif args.preset:
        meshes = resolve_meshes_from_preset(args.preset, args.assets_root)
    elif args.objects:
        meshes = resolve_meshes_from_objects(args.objects, args.assets_dir)
    else:
        meshes = resolve_meshes(args.mesh_glob)
    
    policies = _parse_policies(args.policies)
    
    if not meshes:
        if args.mesh_list:
            print(f"No meshes found in mesh list: {args.mesh_list}")
        elif args.preset:
            print(f"No meshes found for preset='{args.preset}' under assets_root='{args.assets_root}'")
        elif args.objects:
            print(f"No meshes found for objects='{args.objects}' under assets_dir='{args.assets_dir}'")
        else:
            print(f"No meshes found: {args.mesh_glob}")
        return

    agg_results = defaultdict(lambda: {"vrr": [], "vrr_best": [], "success": []})
    output_root = Path(args.output_dir)
    if not args.no_run_subdir:
        output_root = output_root / run_name
    output_root.mkdir(parents=True, exist_ok=True)
    # Ensure figure directory exists before plotting at the end.
    (output_root / "figures").mkdir(parents=True, exist_ok=True)

    # Initialize shared heavy models once
    print("Initializing shared models (Point-E, CLIPSeg)...")
    shared_pointe = PointEGenerator(base_cfg.pointe)
    shared_segmenter = CLIPSegSegmenter(base_cfg.segmentation)

    for mesh_path in meshes:
        mesh_name = Path(mesh_path).stem
        print(f"\n[Mesh] {mesh_name}: {mesh_path}", flush=True)

        # Decide stable-pose normalization for this mesh.
        # - Curated assets: keep original upright orientation (disable stable-pose).
        # - Objaverse assets: enable stable-pose for robustness.
        if normalize_mode == "auto":
            normalize_stable_pose_mesh = _is_objaverse_mesh(mesh_path)
        else:
            normalize_stable_pose_mesh = bool(getattr(base_cfg.simulator, "normalize_stable_pose", True))

        mesh_seed = int(base_cfg.policy.random_seed) + int(args.initial_view_seed) + _stable_hash32(mesh_name)

        # Optional semantic-occlusion pool (computed once per mesh).
        view_pool: list[int] | None = None
        if str(args.initial_view_mode) == "semantic_occluded":
            occl_prompt = args.initial_view_occlusion_prompt or base_cfg.segmentation.prompt
            view_pool = _compute_semantic_occlusion_pool(
                mesh_path=str(mesh_path),
                base_cfg=base_cfg,
                segmenter=shared_segmenter,
                prompt=str(occl_prompt),
                quantile=float(args.initial_view_occlusion_quantile),
                normalize_stable_pose=normalize_stable_pose_mesh,
            )
            if view_pool:
                print(f"[InitView] semantic_occluded pool size={len(view_pool)}/{int(base_cfg.simulator.num_views)} prompt='{occl_prompt}'", flush=True)
        
        for policy in policies:
            for trial in range(args.trials):
                initial_view = _choose_initial_view(
                    mode=str(args.initial_view_mode),
                    fixed_view=int(args.initial_view),
                    trial=int(trial),
                    trials_total=int(args.trials),
                    num_views=int(base_cfg.simulator.num_views),
                    seed=int(mesh_seed),
                    view_pool=view_pool,
                )
                print(
                    f"[Run] Starting {mesh_name} | {policy} | T{trial} | init_view={initial_view}",
                    flush=True,
                )
                cfg = ActiveHallucinationConfig.from_dict(base_cfg.to_dict())
                cfg.simulator.mesh_path = mesh_path
                # Apply per-mesh stable-pose decision.
                cfg.simulator.normalize_stable_pose = bool(normalize_stable_pose_mesh)

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
                
                # Ensure trial-level randomness is consistent across policies:
                # - Random policy view selection uses cfg.policy.random_seed.
                # - Point-E sampling uses cfg.pointe.seed_list when provided.
                #   Without this, Point-E defaults to deterministic seeds [0..K-1],
                #   making multiple trials effectively identical.
                cfg.policy.random_seed = base_cfg.policy.random_seed + trial
                seed_base = base_cfg.policy.random_seed + trial * 1000
                cfg.pointe.seed_list = [seed_base + i for i in range(int(cfg.pointe.num_seeds))]

                # Handle combined score policy variant (e.g., "active_combined")
                actual_policy = policy
                if policy.endswith("_combined"):
                    actual_policy = policy.replace("_combined", "")
                    # Enable combined score (S1 + S2) with a fixed weight for consistency.
                    cfg.variance.semantic_variance_weight = float(args.combined_semantic_variance_weight)
                
                cfg.experiment.policy = actual_policy
                
                cfg.experiment.output_dir = str(output_root / mesh_name)
                cfg.experiment.trajectory_name = f"{policy}_t{trial}"
                
                # Explicitly disable heavy 3D visualization for batch runs if supported
                if hasattr(cfg.experiment, "visualize"):
                    cfg.experiment.visualize = False 

                runner = ActiveHallucinationRunner(
                    cfg, 
                    sim=None, 
                    pointe=shared_pointe, 
                    segmenter=shared_segmenter
                )
                try:
                    result = runner.run_episode(initial_view=int(initial_view))
                    metrics = result["metrics"]
                    agg_results[policy]["vrr"].append(metrics["variance_reduction_rate"])
                    # Monotone alternative (if available).
                    if "variance_reduction_rate_best_so_far" in metrics:
                        agg_results[policy]["vrr_best"].append(metrics["variance_reduction_rate_best_so_far"])
                    agg_results[policy]["success"].append(1.0 if metrics["success_at_final_step"] else 0.0)
                    print(f"[Run] Finished {mesh_name} | {policy} | T{trial}", flush=True)
                except Exception as e:
                    print(f"[Run] Failed {mesh_name} {policy} T{trial}: {e}", flush=True)
                finally:
                    runner.close()

    avg_vrr_map = {}
    avg_vrr_best_map = {}
    for policy, data in agg_results.items():
        vrrs = data["vrr"]
        if not vrrs: continue
        min_len = min(len(v) for v in vrrs)
        mat = np.array([v[:min_len] for v in vrrs])
        avg_vrr_map[policy] = np.mean(mat, axis=0).tolist()

        vrrb = data.get("vrr_best") or []
        if vrrb:
            min_len_b = min(len(v) for v in vrrb)
            mat_b = np.array([v[:min_len_b] for v in vrrb])
            avg_vrr_best_map[policy] = np.mean(mat_b, axis=0).tolist()

    plot_vrr_curves(avg_vrr_map, str(output_root / "figures" / "comparison_vrr.png"))
    if avg_vrr_best_map:
        plot_vrr_curves(
            avg_vrr_best_map,
            str(output_root / "figures" / "comparison_vrr_best_so_far.png"),
        )

    success_map = {}
    for policy, data in agg_results.items():
        outcomes = data["success"]
        success_map[policy] = float(np.mean(outcomes)) if outcomes else 0.0

    plot_success_rates(success_map, str(output_root / "figures" / "comparison_success.png"))
    print(f"Batch run complete. Outputs in {output_root}")


if __name__ == "__main__":
    main()
