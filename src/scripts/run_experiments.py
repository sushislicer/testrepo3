"""Batch runner for multiple meshes and policies."""

from __future__ import annotations

import os
# CRITICAL FIX: Set headless rendering backend before importing visualization libs
# This prevents Open3D/GLFW from trying to open a window and timing out.
os.environ["PYOPENGL_PLATFORM"] = "egl"

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
    return parser.parse_args()


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
        # Matches files like:
        #   assets/ACE_Coffee_Mug_*.obj
        #   assets/meshes/coffee_mug.obj
        # and any other OBJ whose filename contains 'mug'.
        out: list[str] = []
        for p in root.rglob("*.obj"):
            if "mug" in p.name.lower():
                out.append(str(p))
        return sorted(set(out))

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

    agg_results = defaultdict(lambda: {"vrr": [], "success": []})
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    # Ensure figure directory exists before plotting at the end.
    (output_root / "figures").mkdir(parents=True, exist_ok=True)

    for mesh_path in meshes:
        mesh_name = Path(mesh_path).stem
        for policy in policies:
            for trial in range(args.trials):
                cfg = ActiveHallucinationConfig.from_dict(base_cfg.to_dict())
                cfg.simulator.mesh_path = mesh_path
                
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
                    cfg.variance.semantic_variance_weight = 0.5
                
                cfg.experiment.policy = actual_policy
                
                cfg.experiment.output_dir = str(output_root / mesh_name)
                cfg.experiment.trajectory_name = f"{policy}_t{trial}"
                
                # Explicitly disable heavy 3D visualization for batch runs if supported
                if hasattr(cfg.experiment, "visualize"):
                    cfg.experiment.visualize = False 

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
