"""
Distributed runner for Active-Hallucination experiments.
Splits meshes across available GPUs and runs run_experiments.py in parallel.
"""
import argparse
import math
import os
import subprocess
import sys
import time
import shutil
from datetime import datetime
from glob import glob
from pathlib import Path
import torch

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent

def get_available_gpus():
    if torch.cuda.is_available():
        return torch.cuda.device_count()
    return 0

def parse_args():
    parser = argparse.ArgumentParser(description="Run distributed experiments.")
    parser.add_argument("--config", type=str, default=None, help="Path to base YAML config passed to run_experiments.")
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
        "--dry-run",
        action="store_true",
        help="Print commands and mesh lists without executing them.",
    )
    parser.add_argument(
        "--assets_root",
        type=str,
        default="assets",
        help="Project-relative assets root used by presets.",
    )
    parser.add_argument(
        "--mesh_glob",
        type=str,
        default="assets/meshes/**/*.obj",
        help="Glob pattern for meshes (used when --objects is not set).",
    )
    parser.add_argument(
        "--objects",
        type=str,
        default=None,
        help=(
            "Object categories to run (comma-separated, e.g. 'mug,pitcher'), or 'all'. "
            "If provided, meshes are resolved under --assets_dir and override --mesh_glob."
        ),
    )
    parser.add_argument(
        "--assets_dir",
        type=str,
        default="assets/meshes",
        help="Root directory containing category subfolders for --objects.",
    )
    parser.add_argument("--num_gpus", type=int, default=None, help="Number of GPUs to use. Defaults to all available GPUs.")
    parser.add_argument("--output_dir", type=str, default="outputs/distributed_results", help="Base output directory.")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["active", "active_combined", "random", "geometric"],
        help=(
            "Policies to run. Accepts space-separated (recommended) and/or comma-separated values. "
            "Example: --policies active random geometric active_combined"
        ),
    )
    parser.add_argument("--trials", type=int, default=3, help="Trials per setting.")
    return parser.parse_args()

def resolve_meshes(glob_pattern: str) -> list[str]:
    files = sorted(glob(glob_pattern))
    if files: return files
    root_pattern = str(PROJECT_ROOT / glob_pattern)
    return sorted(glob(root_pattern))


def resolve_meshes_from_preset(preset: str, assets_root: str) -> list[str]:
    preset = (preset or "").strip().lower()
    root = Path(assets_root)
    if not root.is_absolute():
        root = PROJECT_ROOT / root

    if preset == "mugs5":
        out: list[str] = []
        for p in root.rglob("*.obj"):
            if "mug" in p.name.lower():
                out.append(str(p))
        return sorted(set(out))

    if preset == "objaverse_subset":
        # Expected path: assets/objaverse_subset/{category}/*.obj
        # If assets_root is 'assets', then we look in assets/objaverse_subset
        # If assets_root is 'assets/objaverse_subset', we look in .
        
        # We'll assume assets_root points to 'assets' by default.
        # If the user passed a different assets_root, we respect it.
        
        # Check if 'objaverse_subset' exists in root
        subset_root = root / "objaverse_subset"
        if not subset_root.exists():
            # Maybe root IS the subset dir?
            if (root / "mugs").exists():
                subset_root = root
            else:
                # Fallback or error
                pass
        
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


def _parse_policies(policies_tokens: list[str]) -> list[str]:
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

def main():
    args = parse_args()
    
    available_gpus = get_available_gpus()
    if args.num_gpus is None:
        args.num_gpus = available_gpus
        print(f"Auto-detected {available_gpus} GPUs. Using all of them.")
    
    if args.num_gpus <= 0:
        # Fallback to 1 if no GPUs found (maybe CPU run? but code seems to expect GPU)
        # Or just raise error if GPU is required.
        # Assuming GPU is required for now based on context.
        if available_gpus == 0:
             print("Warning: No GPUs detected. Setting num_gpus=1 (might fail if CUDA is required).")
             args.num_gpus = 1
        else:
             raise ValueError("--num_gpus must be > 0")

    if args.num_gpus > available_gpus:
        print(f"Warning: Requested {args.num_gpus} GPUs but only {available_gpus} are available.")
        print(f"Clamping num_gpus to {available_gpus}.")
        args.num_gpus = available_gpus

    # If running inside a scheduler/ssh environment, be explicit about a safe
    # multiprocessing start method for subprocesses (we don't use torch.distributed).
    # No-op here; kept as documentation.
    
    # 1. Find meshes
    if args.preset:
        meshes = resolve_meshes_from_preset(args.preset, args.assets_root)
    elif args.objects:
        meshes = resolve_meshes_from_objects(args.objects, args.assets_dir)
    else:
        meshes = resolve_meshes(args.mesh_glob)
    if not meshes:
        if args.preset:
            print(f"No meshes found for preset='{args.preset}' under assets_root='{args.assets_root}'")
        elif args.objects:
            print(f"No meshes found for objects='{args.objects}' under assets_dir='{args.assets_dir}'")
        else:
            print(f"No meshes found matching: {args.mesh_glob}")
        sys.exit(1)
        
    print(f"Found {len(meshes)} meshes. Distributing across {args.num_gpus} GPUs.")
    
    # 2. Split meshes
    chunk_size = math.ceil(len(meshes) / args.num_gpus)
    chunks = [meshes[i:i + chunk_size] for i in range(0, len(meshes), chunk_size)]
    
    # Ensure we have exactly num_gpus chunks (some might be empty if meshes < num_gpus)
    while len(chunks) < args.num_gpus:
        chunks.append([])
        
    # 3. Create temp files and launch processes
    processes = []
    temp_files = []
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"run_{timestamp}"
    
    # Create output dir
    output_path = Path(args.output_dir) / run_id
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Output directory: {output_path}")
    
    policies = _parse_policies(args.policies)

    for gpu_id, mesh_chunk in enumerate(chunks):
        if not mesh_chunk:
            print(f"GPU {gpu_id}: No meshes assigned.")
            continue
            
        # Create temp list file
        list_file = output_path / f"mesh_list_gpu{gpu_id}.txt"
        with open(list_file, "w") as f:
            for m in mesh_chunk:
                f.write(f"{m}\n")
        temp_files.append(list_file)
        
        # Construct command
        cmd = [
            sys.executable, "-m", "src.scripts.run_experiments",
            "--mesh_list", str(list_file),
        ]
        if args.config:
            cmd += ["--config", str(args.config)]
        if args.preset:
            cmd += ["--preset", str(args.preset), "--assets_root", str(args.assets_root)]
        cmd += [
            "--policies",
            *policies,
            "--trials",
            str(args.trials),
            "--output_dir",
            str(output_path),
        ]
        
        env = os.environ.copy()
        # Constrain each worker to a single visible GPU. This script uses
        # process-level parallelism (not torch.distributed), so this is the
        # correct isolation mechanism on multi-GPU nodes.
        
        # If we have fewer GPUs than chunks (shouldn't happen with logic above unless forced),
        # we might oversubscribe.
        # But we clamped args.num_gpus to available_gpus.
        # So gpu_id will be 0..available_gpus-1.
        
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Headless rendering compatibility on clusters (EGL).
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        env.setdefault("EGL_PLATFORM", "surfaceless")
        # On some cluster/VM images with GLVND, EGL may accidentally select Mesa.
        # Pin the NVIDIA vendor JSON when available to avoid driver mismatches.
        nvidia_vendor_json = "/usr/share/glvnd/egl_vendor.d/10_nvidia.json"
        if "__EGL_VENDOR_LIBRARY_FILENAMES" not in env and os.path.exists(nvidia_vendor_json):
            env["__EGL_VENDOR_LIBRARY_FILENAMES"] = nvidia_vendor_json
        # Reduce CPU over-subscription when launching N workers.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        
        print(f"Launching GPU {gpu_id} with {len(mesh_chunk)} meshes...")
        if args.dry_run:
            print(f"[DRY RUN] Command: {' '.join(cmd)}")
            # We don't launch the process
        else:
            p = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT))
            processes.append(p)
        
    # 4. Wait for completion
    if args.dry_run:
        print("Dry run complete. No processes launched.")
        return

    print("Waiting for processes to complete...")
    exit_codes = [p.wait() for p in processes]
    
    if any(code != 0 for code in exit_codes):
        print("Some processes failed.")
    else:
        print("All processes completed successfully.")
        
    # 5. Archive results
    export_dir = PROJECT_ROOT / "results_export"
    export_dir.mkdir(exist_ok=True)
    
    archive_name = export_dir / f"results_{run_id}"
    print(f"Archiving results to {archive_name}.zip ...")
    
    shutil.make_archive(str(archive_name), 'zip', str(output_path))
    
    print(f"Done! Results are ready at: {archive_name}.zip")
    print("You can now git add/commit this file or download it.")

if __name__ == "__main__":
    main()
