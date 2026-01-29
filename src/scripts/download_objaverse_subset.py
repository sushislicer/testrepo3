"""Download a small real-mesh subset (Objaverse) into `assets/objaverse_subset`.

This script targets Objaverse via the official `objaverse` Python package and
filters objects using LVIS annotations for faster and more reliable category matching.

Why this exists:
- Objaverse is huge; you usually want a small subset for evaluation.
- We want a repeatable way to populate `assets/<category>/...`.

Prereqs (run once in your environment):
- `pip install objaverse trimesh`

Example:
  python3 -m src.scripts.download_objaverse_subset \
    --categories mugs,pans,kettles,hammers \
    --max_per_category 50 \
    --out_dir assets/objaverse_subset \
    --export_obj

Notes:
- Objaverse terms/licensing: respect dataset + upstream asset licenses.
- This script can consume significant disk space; start with small limits.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent


def _normalize_mesh_to_sim_convention(mesh, *, stable_pose: bool = True):
    """Normalize mesh into a tabletop-friendly convention.

    Goals:
    - apply a plausible *upright/stable* orientation (Objaverse assets have inconsistent up-axes)
    - center in XY
    - put the lowest point on z=0

    The stable-pose step is best-effort and falls back to translation-only.
    """
    import numpy as np

    try:
        if stable_pose:
            # Try trimesh stable-pose estimation (works well for mugs/pans/etc).
            # API varies slightly across trimesh versions, so keep this very defensive.
            poses = None
            probs = None
            try:
                poses, probs = mesh.compute_stable_poses()  # type: ignore[attr-defined]
            except Exception:
                try:
                    from trimesh.poses import compute_stable_poses

                    poses, probs = compute_stable_poses(mesh)
                except Exception:
                    poses, probs = None, None

            if poses is not None and len(poses) > 0:
                best_i = 0
                if probs is not None and len(probs) == len(poses):
                    try:
                        best_i = int(np.argmax(np.asarray(probs, dtype=np.float32)))
                    except Exception:
                        best_i = 0
                T = np.asarray(poses[best_i], dtype=np.float32)
                if T.shape == (4, 4):
                    mesh.apply_transform(T)
    except Exception:
        # Never hard-fail the downloader due to pose estimation.
        pass

    # Center XY and put base on z=0.
    bounds = np.asarray(mesh.bounds, dtype=np.float32)
    min_xyz, max_xyz = bounds
    translate = np.array(
        [-0.5 * (min_xyz[0] + max_xyz[0]), -0.5 * (min_xyz[1] + max_xyz[1]), -float(min_xyz[2])],
        dtype=np.float32,
    )
    mesh.apply_translation(translate)
    return mesh


def _prune_dir_to_obj_only(dir_path: Path) -> None:
    """Remove non-.obj files and nested directories under a category folder."""
    for p in list(dir_path.rglob("*")):
        if p.is_dir():
            continue
        if p.suffix.lower() == ".obj":
            continue
        try:
            p.unlink()
        except Exception:
            pass
    # Remove now-empty directories.
    for d in sorted([p for p in dir_path.rglob("*") if p.is_dir()], reverse=True):
        try:
            d.rmdir()
        except Exception:
            pass


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download an Objaverse subset into assets/")
    p.add_argument(
        "--categories",
        type=str,
        default="mugs,pans,kettles,hammers",
        help="Comma-separated category list: mugs,pans,kettles,hammers",
    )
    p.add_argument(
        "--max_per_category",
        type=int,
        default=50,
        help="Maximum number of objects to download per category.",
    )
    p.add_argument(
        "--out_dir",
        type=str,
        default="assets/objaverse_subset",
        help="Output directory (project-relative or absolute).",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Shuffle seed used to sample objects.",
    )
    p.add_argument(
        "--download_processes",
        type=int,
        default=8,
        help="Parallelism for downloading objects (passed to objaverse, if supported).",
    )
    p.add_argument(
        "--export_obj",
        action="store_true",
        help="Convert downloaded meshes to .obj under out_dir/<category>/.",
    )
    p.add_argument(
        "--keep_original",
        action="store_true",
        help="Keep original downloaded files (e.g., .glb) alongside exported .obj.",
    )
    p.add_argument(
        "--keep_aux_files",
        action="store_true",
        help=(
            "Keep auxiliary non-mesh files in category folders (textures/materials/etc). "
            "By default, we prune non-.obj files to avoid loading issues and keep the subset clean."
        ),
    )
    p.add_argument(
        "--no_stable_pose",
        action="store_true",
        help="Disable stable-pose upright normalization during OBJ export.",
    )
    p.add_argument(
        "--dry_run",
        action="store_true",
        help="Only print how many objects would be selected; do not download.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Map our categories to LVIS categories
    # Keys are our category names, Values are lists of LVIS category names to include
    cat_mapping = {
        "mugs": ["mug"],
        "pans": ["frying_pan", "saucepan", "pan_(for_cooking)"],
        "kettles": ["kettle", "teakettle"],
        "hammers": ["hammer"],
    }

    requested = [c.strip().lower() for c in str(args.categories).split(",") if c.strip()]
    unknown = [c for c in requested if c not in cat_mapping]
    if unknown:
        raise SystemExit(f"Unknown categories: {unknown}. Available: {sorted(cat_mapping.keys())}")

    # Import lazily so the repo remains usable without objaverse installed.
    try:
        import objaverse
    except Exception as e:  # pragma: no cover
        raise SystemExit(
            "Missing dependency 'objaverse'. Install with: pip install objaverse\n"
            f"Import error: {type(e).__name__}: {e}"
        )

    print("Loading Objaverse LVIS annotations...")
    # Use LVIS annotations which are a subset and much faster/lighter than full annotations
    lvis_anns = objaverse.load_lvis_annotations()
    
    rng = random.Random(int(args.seed))
    
    selected_by_cat: Dict[str, List[str]] = {c: [] for c in requested}
    max_n = int(args.max_per_category)

    for c in requested:
        if c not in cat_mapping:
            # Should be caught above, but safe check
            continue
            
        lvis_cats = cat_mapping[c]
        candidates = []
        for lc in lvis_cats:
            if lc in lvis_anns:
                candidates.extend(lvis_anns[lc])
        
        # Remove duplicates if any
        candidates = list(set(candidates))
        rng.shuffle(candidates)
        
        selected_by_cat[c] = candidates[:max_n]

    print("Selection summary:")
    for c in requested:
        print(f"  {c}: {len(selected_by_cat[c])}")

    manifest = {
        "source": "objaverse",
        "categories": {c: selected_by_cat[c] for c in requested},
        "seed": int(args.seed),
        "max_per_category": int(args.max_per_category),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest: {out_dir / 'manifest.json'}")

    if args.dry_run:
        print("Dry run complete.")
        return

    # Download per category so you can re-run/resume categories independently.
    all_downloaded: Dict[str, Dict[str, str]] = {}
    for c in requested:
        cat_dir = out_dir / c
        cat_dir.mkdir(parents=True, exist_ok=True)
        cat_uids = selected_by_cat[c]
        if not cat_uids:
            continue

        print(f"Downloading {len(cat_uids)} objects for '{c}'...")
        kwargs = {}
        # Some objaverse versions accept `download_processes` and/or `output_dir`.
        # Use best-effort kwargs to stay compatible across versions.
        kwargs["uids"] = cat_uids
        if "download_processes" in objaverse.load_objects.__code__.co_varnames:  # type: ignore[attr-defined]
            kwargs["download_processes"] = int(args.download_processes)
        if "download_dir" in objaverse.load_objects.__code__.co_varnames:  # type: ignore[attr-defined]
            kwargs["download_dir"] = str(cat_dir)
        if "output_dir" in objaverse.load_objects.__code__.co_varnames:  # type: ignore[attr-defined]
            kwargs["output_dir"] = str(cat_dir)

        downloaded: Dict[str, str] = objaverse.load_objects(**kwargs)
        all_downloaded[c] = downloaded

        if args.export_obj:
            try:
                import trimesh
            except Exception as e:
                raise SystemExit(
                    "Missing dependency 'trimesh'. Install with: pip install trimesh\n"
                    f"Import error: {type(e).__name__}: {e}"
                )

            for uid, src_path in downloaded.items():
                try:
                    src = Path(src_path)
                    if not src.exists():
                        continue
                    mesh = trimesh.load(str(src), force="mesh")
                    if isinstance(mesh, trimesh.Scene):
                        if not mesh.geometry:
                            continue
                        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))
                    mesh = _normalize_mesh_to_sim_convention(mesh, stable_pose=not bool(args.no_stable_pose))
                    dst = cat_dir / f"{uid}.obj"
                    mesh.export(dst)
                    if not args.keep_original:
                        # Keep non-OBJ originals by default unless requested.
                        if src.suffix.lower() != ".obj":
                            try:
                                src.unlink()
                            except Exception:
                                pass
                except Exception as e:
                    print(f"[WARN] Export failed for {uid}: {type(e).__name__}: {e}")

            # Optional pruning: keep only *.obj in each category folder.
            if not args.keep_aux_files and not args.keep_original:
                _prune_dir_to_obj_only(cat_dir)

    (out_dir / "downloaded_paths.json").write_text(json.dumps(all_downloaded, indent=2), encoding="utf-8")
    print(f"Wrote downloaded paths: {out_dir / 'downloaded_paths.json'}")
    print("Done.")


if __name__ == "__main__":
    main()
