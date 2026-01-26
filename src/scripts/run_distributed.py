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

CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent

def parse_args():
    parser = argparse.ArgumentParser(description="Run distributed experiments.")
    parser.add_argument("--mesh_glob", type=str, default="assets/meshes/*.obj", help="Glob pattern for meshes.")
    parser.add_argument("--num_gpus", type=int, default=4, help="Number of GPUs to use.")
    parser.add_argument("--output_dir", type=str, default="outputs/distributed_results", help="Base output directory.")
    parser.add_argument("--policies", type=str, default="active,active_combined,random,geometric", help="Policies to run.")
    parser.add_argument("--trials", type=int, default=3, help="Trials per setting.")
    return parser.parse_args()

def resolve_meshes(glob_pattern: str) -> list[str]:
    files = sorted(glob(glob_pattern))
    if files: return files
    root_pattern = str(PROJECT_ROOT / glob_pattern)
    return sorted(glob(root_pattern))

def main():
    args = parse_args()
    if args.num_gpus <= 0:
        raise ValueError("--num_gpus must be > 0")
    # If running inside a scheduler/ssh environment, be explicit about a safe
    # multiprocessing start method for subprocesses (we don't use torch.distributed).
    # No-op here; kept as documentation.
    
    # 1. Find meshes
    meshes = resolve_meshes(args.mesh_glob)
    if not meshes:
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
            "--policies", args.policies,
            "--trials", str(args.trials),
            "--output_dir", str(output_path)
        ]
        
        env = os.environ.copy()
        # Constrain each worker to a single visible GPU. This script uses
        # process-level parallelism (not torch.distributed), so this is the
        # correct isolation mechanism on multi-GPU nodes.
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
        # Headless rendering compatibility on clusters (EGL).
        env.setdefault("PYOPENGL_PLATFORM", "egl")
        # Reduce CPU over-subscription when launching N workers.
        env.setdefault("OMP_NUM_THREADS", "1")
        env.setdefault("MKL_NUM_THREADS", "1")
        
        print(f"Launching GPU {gpu_id} with {len(mesh_chunk)} meshes...")
        p = subprocess.Popen(cmd, env=env, cwd=str(PROJECT_ROOT))
        processes.append(p)
        
    # 4. Wait for completion
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
