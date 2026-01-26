"""
Smoke test script to verify the pipeline on a single object.
Runs:
1. Standard suite: Active (S1), Random, Geometric.
2. Combined suite: Active (S1+S2).
"""
import sys
import os
import yaml
from pathlib import Path

# Add project root to path
CURRENT_SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = CURRENT_SCRIPT_PATH.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from src.scripts.run_experiments import main as run_experiments_main

def main():
    mesh_path = "assets/meshes/coffee_mug.obj"
    base_output_dir = PROJECT_ROOT / "outputs" / "smoke_test"
    
    if not (PROJECT_ROOT / mesh_path).exists():
        print(f"Warning: {mesh_path} not found. Please ensure assets are present.")
        return

    print(f"--- Phase 1: Standard Comparison (Active S1, Random, Geometric) ---")
    sys.argv = [
        "run_experiments.py",
        "--mesh_glob", mesh_path,
        "--policies", "active,random,geometric",
        "--trials", "1",
        "--output_dir", str(base_output_dir / "standard")
    ]
    try:
        run_experiments_main()
    except Exception as e:
        print(f"Phase 1 Failed: {e}")
        sys.exit(1)

    print(f"\n--- Phase 2: Combined Score (Active S1+S2) ---")
    # Create temp config
    config_path = base_output_dir / "combined_config.yaml"
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    combined_config = {
        "variance": {
            "semantic_variance_weight": 0.5
        }
    }
    
    with open(config_path, "w") as f:
        yaml.dump(combined_config, f)
        
    sys.argv = [
        "run_experiments.py",
        "--config", str(config_path),
        "--mesh_glob", mesh_path,
        "--policies", "active",
        "--trials", "1",
        "--output_dir", str(base_output_dir / "combined")
    ]
    
    try:
        run_experiments_main()
    except Exception as e:
        print(f"Phase 2 Failed: {e}")
        sys.exit(1)
        
    print("\nSmoke test completed!")
    print(f"Standard results: {base_output_dir}/standard")
    print(f"Combined results: {base_output_dir}/combined")

if __name__ == "__main__":
    main()
