# Active-Hallucination

Simulation-only pipeline that leverages generative variance (Point-E) plus CLIPSeg-based semantic *painting* of each Point‑E hypothesis (via multiview point-cloud renders) to drive Next-Best-View selection for revealing object affordances (e.g., handles) in tabletop scenes.

## Quickstart

1) Create environment (example):
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Ensure GPU-enabled PyTorch if available (`python - <<'PY'\nimport torch; print(torch.cuda.is_available())\nPY`).
3) Put meshes under `assets/`.
   - This repo already includes 5 mug meshes directly under `assets/` (e.g. `assets/ACE_Coffee_Mug_Kristen_16_oz_cup.obj`).
   - Legacy/demo meshes may also live under `assets/meshes/`.
   - Supported by the loader: OBJ/PLY/GLB/STL (the default glob targets OBJ).
   - The simulator normalizes meshes to sit on the table (base on `z=0`) and centers them in `x/y`.
4) Run a dry demo (no heavy compute) to verify rendering + imports:
```
python -m src.scripts.demo_orbital_camera --mesh assets/meshes/example.obj
```

## Layout

- `src/`: core library (importable as the `src` package in this repo)
  - `config.py`: dataclasses + YAML helpers.
  - `simulator.py`: pyrender-based tabletop camera + renderer.
  - `pointe_wrapper.py`: Point-E multi-seed generation helper with graceful fallbacks.
  - `variance_field.py`: voxel grid + variance/semantic score computation.
  - `segmentation.py`: CLIPSeg semantic masking + multiview 3D painting helpers.
  - `nbv_policy.py`: Active-Hallucination NBV + baselines.
  - `experiments.py`: end-to-end loops, metrics, logging.
  - `visualization.py`: overlays, variance heatmaps, trajectory grids.
- `src/scripts/`: CLI entrypoints for demos and experiments.
- `assets/`: mesh assets (this repo includes 5 mugs at `assets/*.obj`), plus optional subfolders like `assets/meshes/`.
- `outputs/`: renders, logs, and figures.
- `configs/`: YAML configs for experiments.

## Scripts (CLI entrypoints)

All entrypoints live under `src/scripts/` and can be run via `python -m src.scripts.<name> ...`.

- `demo_orbital_camera`: sanity-check the simulator by rendering an orbital ring of RGB views for a single mesh.
  - Inputs: `--mesh <path>`, optional `--num_views <N>`, `--output <dir>`
  - Outputs: `view_00.png`, `view_01.png`, ... under `--output` (default `outputs/demo_orbit`)
  - How it works: constructs a default `ActiveHallucinationConfig`, overrides `cfg.simulator.mesh_path`, initializes `VirtualTabletopSimulator`, then calls `sim.render_view(i)` for `i=0..num_views-1` and writes each RGB frame via `imageio`.
  - Example: `python -m src.scripts.demo_orbital_camera --mesh assets/meshes/example.obj --num_views 24`

- `demo_pointe_multiseed`: sample multiple Point-E reconstructions (different random seeds) from a single RGB image.
  - Inputs: `--image <path>`, optional `--prompt <text>`, `--num_seeds <N>`, `--save_dir <dir>`, `--view`
  - Outputs: `cloud_00.npy`, `cloud_01.npy`, ... under `--save_dir` (default `outputs/demo_pointe`)
  - Notes: `--view` attempts an Open3D overlay visualization (if Open3D is installed and working).
  - How it works: loads the image with PIL, calls `PointEGenerator.generate_point_clouds_from_image(...)` to produce multiple point clouds (one per seed), saves them as `.npy`, and optionally visualizes them with `overlay_point_clouds_open3d`.
  - Example: `python -m src.scripts.demo_pointe_multiseed --image path/to/image.png --num_seeds 8 --view`

- `demo_variance_field`: compute a voxelized variance field from Point-E multi-seed outputs, optionally weighted by CLIPSeg semantic painting on each Point‑E generation (3 rendered views per generation).
  - Inputs: `--image <path>`, optional `--prompt <text>`, `--use_mask`, `--save_dir <dir>`
  - Outputs: `variance_projection.png` (max-projection) and `variance_points.npy` (thresholded 3D score points) under `--save_dir` (default `outputs/demo_variance`)
  - Notes: `--use_mask` renders each Point‑E point cloud from multiple viewpoints (default: 3 views at 120° apart), runs CLIPSeg on those renders, and projects the resulting masks back to 3D points to produce semantic weights.
  - How it works: (1) generates multi-seed Point-E point clouds from the input image, (2) voxelizes them into a grid and computes per-voxel variance (`compute_variance_field`), (3) optionally performs multiview CLIPSeg semantic painting per generation and accumulates per-voxel semantic weights (`accumulate_semantic_weights`), then (4) combines variance+semantics (`combine_variance_and_semantics`) and exports a 2D max-projection plus thresholded 3D score points.
  - Example: `python -m src.scripts.demo_variance_field --image path/to/image.png --use_mask`

- `run_single_object_demo`: run one complete Active-Hallucination “episode” (NBV loop) for a single mesh and log results.
  - Inputs: `--mesh <path>`, optional `--config <yaml>`, `--policy active|random|geometric`, `--steps <N>`, `--initial_view <idx>`, `--output <dir>`
  - Outputs: per-step renders/figures/logs under `cfg.experiment.output_dir/cfg.experiment.trajectory_name` (defaults are set in `src/config.py`)
  - How it works: loads a base config (optionally from YAML), applies CLI overrides, then runs `ActiveHallucinationRunner.run_episode(...)` which iterates for `num_steps`: render a view, run Point-E multi-seed generation, multiview-render each Point‑E generation and run CLIPSeg to paint affordance weights in 3D, compute the variance×semantic score volume, select the next view using the chosen policy, and log artifacts to the trajectory folder.
  - Example: `python -m src.scripts.run_single_object_demo --mesh assets/meshes/example.obj --policy active --steps 8`

- `run_experiments`: batch runner over many meshes and policies (useful for quick comparisons).
  - Inputs:
    - `--config <yaml>` (optional)
    - One of:
      - `--preset mugs5` (recommended for this repo; runs all OBJ meshes under `assets/` whose filename contains `mug`)
      - `--mesh_glob <glob>`
      - `--mesh_list <txt>`
      - `--objects <comma,categories>` + `--assets_dir <dir>` (category folders)
    - `--policies ...` (space-separated recommended; comma-separated also supported)
    - `--trials <N>`
  - Outputs: one trajectory folder per `(mesh, policy, trial)` under the configured output directory
  - How it works: expands `--mesh_glob`, loops over meshes × policies × trials, clones the base config each time, sets `trajectory_name` (and offsets `random_seed` by trial), then calls `ActiveHallucinationRunner.run_episode()` for each run.
  - Examples:
    - Evaluate the 5 mug meshes shipped in this repo:
      - `python3 -m src.scripts.run_experiments --config configs/main.yaml --preset mugs5 --policies active active_combined random geometric --trials 5`
    - Run all OBJ meshes under `assets/meshes/`:
      - `python3 -m src.scripts.run_experiments --mesh_glob 'assets/meshes/**/*.obj' --policies active random geometric --trials 3`

- `smoke_test`: quick verification script to ensure the pipeline runs end-to-end on a single object.
  - Inputs: None (hardcoded to use `assets/meshes/coffee_mug.obj` if present).
  - Outputs: `outputs/smoke_test/standard` and `outputs/smoke_test/combined`.
  - How it works: Runs two phases. Phase 1 runs the standard suite (Active S1, Random, Geometric). Phase 2 runs the Combined Score configuration (Active S1+S2) to verify the integration of semantic variance weights.
  - Example: `python -m src.scripts.smoke_test`

- `run_distributed`: Distributed runner for multi-GPU setups (e.g., 4x A800).
  - Inputs: `--num_gpus <N>`, `--policies ...`, `--trials <N>`, plus one of `--preset/--objects/--mesh_glob`.
  - Outputs: Zipped results in `results_export/results_<timestamp>.zip`.
  - How it works: Splits the meshes into N chunks and launches `run_experiments` on each GPU in parallel. Automatically archives the results for easy export.
  - Note: Default policies include `active_combined`, which tests the Active policy with the Combined Score (S1+S2) enabled.
  - Examples:
    - 4×GPU evaluation on the repo’s 5 mug meshes:
      - `python3 -m src.scripts.run_distributed --num_gpus 4 --config configs/main.yaml --preset mugs5 --policies active active_combined random geometric --trials 5`
    - Multi-GPU run over a mesh glob:
      - `python3 -m src.scripts.run_distributed --num_gpus 4 --mesh_glob "assets/meshes/**/*.obj" --policies active active_combined random geometric --trials 5`

## Notes

- Point-E and CLIPSeg weights are downloaded automatically via HuggingFace when first used. Expect GPU to speed up Point-E; CPU will be slow.
- The pipeline is inference-only; no training loops are included.
- See `documents/plan.md` for milestone guidance and `documents/project.md` for the research pitch.

## Cluster Usage & Result Export

### Environment
The code is compatible with the `registry.baidubce.com/inference/aibox-pytorch:v1.0-torch2.5.1-cu12.4` Docker image. Ensure `requirements.txt` dependencies are installed.

### Running Distributed Experiments
To utilize multiple GPUs (e.g., 4x A800) efficiently:

```bash
# Example: run the 5 mug meshes included in this repo on 4 GPUs
python3 -m src.scripts.run_distributed --num_gpus 4 --config configs/main.yaml --preset mugs5 \
  --policies active active_combined random geometric --trials 5
```

### Obtaining Meshes (required for distributed runs)

The distributed runner expands a mesh selection (via `--preset`, `--objects`, or `--mesh_glob`) and splits the resulting files across GPUs. Before running on a cluster, ensure you have **at least one mesh** available.

#### Option B: Systematically download *real* meshes (Objaverse)

If you want many real-world meshes for categories like **mugs / pans / kettles / hammers**, the most practical route is **Objaverse**.

This repo includes a small helper script that uses the official `objaverse` Python package to:
- load Objaverse annotations
- filter objects by simple keyword matching (e.g., “mug”, “kettle”, …)
- download a bounded number per category into `assets/objaverse/<category>/...`
- (optional) convert the downloaded assets to `.obj` and normalize them to the simulator convention (centered, base on `z=0`).

1) Install dependencies (in your venv/conda env):
```bash
pip install objaverse trimesh
```

2) Download a small subset into `assets/objaverse/`:
```bash
python3 -m src.scripts.download_objaverse_subset \
  --categories mugs,pans,kettles,hammers \
  --max_per_category 50 \
  --out_dir assets/objaverse \
  --export_obj
```

3) Run distributed experiments over the downloaded meshes:
```bash
python3 -m src.scripts.run_distributed \
  --num_gpus 4 \
  --mesh_glob "assets/objaverse/**/*.obj" \
  --policies active active_combined random geometric \
  --trials 5
```

**Important:** Objaverse is a dataset of assets aggregated from multiple sources. Make sure you comply with the dataset terms and the individual asset licenses.

#### Option A: Use your own meshes
1. Create a directory (either is fine):
   - `assets/` (flat layout)
   - `assets/meshes/` (subfolder)
2. Copy meshes into it (supported by the loader: OBJ/PLY/GLB/STL; the default glob targets OBJ).
3. Recommended conventions for best results:
   - Mesh is roughly **centered near the origin** and the base sits on **`z=0`**.
   - Mesh is **watertight** or reasonably clean (fewer self-intersections helps rendering).
   - If you use non-OBJ formats, pass a matching glob, e.g.:
     - `--mesh_glob "assets/meshes/*.ply"`

#### Option B: Download sample meshes from public datasets
You can populate `assets/meshes/` from common public sources (license-dependent):
- **Google Scanned Objects (GSO)**: high-quality object scans.
- **ShapeNet**: large collection of CAD models.
- **Objaverse / Objaverse-XL**: broad collection of 3D assets.

After downloading, export/convert models to `.obj` (or adjust `--mesh_glob`) and place them under `assets/` (or `assets/meshes/`).

#### Quick sanity check
Run a single-mesh render demo first to verify rendering works in your environment:
```bash
python -m src.scripts.demo_orbital_camera --mesh assets/meshes/<your_mesh>.obj
```

This script will:
1.  Split the meshes across available GPUs.
2.  Run experiments in parallel.
3.  Archive the results into `results_export/results_<timestamp>.zip`.

### Exporting Results via Git
Since the cluster environment may be headless, you can export results by committing the generated zip file:

1.  Run the distributed script (wait for completion).
2.  Add the generated zip file:
    ```bash
    git add results_export/*.zip
    ```
3.  Commit and push:
    ```bash
    git commit -m "Add experiment results"
    git push
    ```
