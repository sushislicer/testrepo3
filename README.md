# Active-Hallucination

Simulation-only pipeline that leverages generative variance (Point-E) plus CLIPSeg-based semantic *painting* of each Point‑E hypothesis (via multiview point-cloud renders) to drive Next-Best-View selection for revealing object affordances (e.g., handles) in tabletop scenes.

## Quickstart

1) **Install System Dependencies** (required for rendering):
   ```bash
   apt-get update && apt-get install -y libgl1 libglib2.0-0 libglu1-mesa libegl1 libgles2
   ```

2) **Create Environment**:
   ```bash
   # Ensure you use --system-site-packages if your base image has Torch/CUDA installed
   python3 -m venv .venv --system-site-packages
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
   *Note: If you are in China, you may need to manually download Point-E weights and CLIPSeg models due to network blocks. See "Troubleshooting" below.*

3) **Run Distributed Experiment**:
   ```bash
   # Example: Run the objaverse subset on 2 GPUs
   HF_ENDPOINT=https://hf-mirror.com python3 -m src.scripts.run_distributed \
     --preset objaverse_subset \
     --num_gpus 2 \
     --policies active active_combined random geometric \
     --trials 5
   ```

4) **Verify Results**:
    Results are archived in `results_export/`.

5) **Aggregate results into tables (CSV)**:
   ```bash
   # Point this at a run folder that contains many <mesh>/<trajectory>/trajectory.json files.
   python3 -m src.scripts.aggregate_results \
     --input_dir outputs/batch_results/run_YYYYMMDD_HHMMSS
   ```
   This writes:
   - `figures/summary_runs.csv` (one row per mesh/policy/trial)
   - `figures/summary_by_policy.csv` (mean/std per policy, plus semantic pass@k)

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
  - This script now supports *run_experiments-like* options: multiple policies, multiple trials, and initial-view selection modes.
  - Inputs:
    - `--mesh <path>` (required), optional `--config <yaml>`
    - `--policies ...` (e.g. `active active_combined random geometric`)
    - `--trials <N>`
    - `--initial_view_mode fixed|random_per_trial|sweep|semantic_occluded` (+ related flags)
    - Combined-score knobs: `--combined_semantic_variance_weight`, `--semantic_threshold`, `--semantic_gamma`, `--semantic_s2_*`
    - Output: `--output_dir`, optional `--run_name`, `--no_run_subdir`
  - Outputs: one trajectory folder per `(policy, trial)` under:
    - `--output_dir/run_YYYYMMDD_HHMMSS/<mesh_name>/<policy>_t<trial>/...` (default)
  - Example (your requested style):
    - `python3 -m src.scripts.run_single_object_demo --mesh assets/ACE_Coffee_Mug_Kristen_16_oz_cup.obj --policies active_combined --trials 3 --initial_view_mode sweep --semantic_s2_power 0.5`

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
    - (New) Initial view selection:
      - `--initial_view_mode fixed|random_per_trial|sweep|semantic_occluded`
      - `--initial_view <idx>` (used by `fixed`)
      - `--initial_view_seed <int>` (re-sample initial views across repeated runs)
      - `--initial_view_occlusion_prompt <text>` (used by `semantic_occluded`, default: segmentation prompt)
      - `--initial_view_occlusion_quantile <q>` (bottom-q most occluded semantic views; default 0.25)
    - (New) Combined-score tuning knobs (apply to `*_combined` policies):
      - `--combined_semantic_variance_weight <float>` (replaces the previous hardcoded 0.5)
      - `--semantic_threshold <float>` and `--semantic_gamma <float>` (override semantic gating/sharpening)
  - Outputs: one trajectory folder per `(mesh, policy, trial)` under the configured output directory
  - How it works: expands `--mesh_glob`, loops over meshes × policies × trials, clones the base config each time, sets `trajectory_name` (and offsets `random_seed` by trial), then calls `ActiveHallucinationRunner.run_episode()` for each run.
  - Examples:
    - Evaluate the 5 mug meshes shipped in this repo:
      - `python3 -m src.scripts.run_experiments --config configs/main.yaml --preset mugs5 --policies active active_combined random geometric --trials 5`
    - Run all OBJ meshes under `assets/meshes/`:
      - `python3 -m src.scripts.run_experiments --mesh_glob 'assets/meshes/**/*.obj' --policies active random geometric --trials 3`

  ### Run output folders (new)

  By default, `run_experiments` now creates a timestamped subfolder under `--output_dir` so repeated runs don't overwrite each other:

  - `--output_dir outputs/batch_results` produces: `outputs/batch_results/run_YYYYMMDD_HHMMSS/...`

  Control this behavior via:
  - `--run_name <name>`: set a custom run folder name.
  - `--no_run_subdir`: write directly into `--output_dir` (legacy behavior; used internally by `run_distributed`).

  ### Combined policy semantics (updated)

  `*_combined` enables a **combined 3D score field**:

  - `S1 = Var(Geometry) * Mean(Semantics)`
  - `S2 = Var(Semantics)` (disagreement across seeds)
  - `score = S1 + w * S2` via [`combine_variance_and_semantics()`](src/variance_field.py:123)

  In addition, NBV selection for `*_combined` now scores candidate views against the **distribution of top‑k score voxels** (not only a single centroid), so semantic signal can influence NBV more directly:
  - top‑k voxel extraction with scores: [`export_topk_score_points()`](src/variance_field.py:199)
  - weighted NBV selection: [`select_next_view_active_weighted()`](src/nbv_policy.py:59)

  #### S2 tuning knobs

  Besides `--combined_semantic_variance_weight`, you can keep S2 alive when mean semantic confidence is low:
  - `--semantic_s2_threshold` (softer gate than S1)
  - `--semantic_s2_gamma`
  - `--semantic_s2_power` (default 0.5 boosts small S2 values)

  #### Semantic visibility + pass@k (new metrics)

  Each step now logs and saves a simple view-based semantic diagnostic:
  - `semantic_visibility`: fraction of simulator RGB pixels above the CLIPSeg threshold for the affordance prompt.
  - debug image: `step_XX_rgb_semantic_mask.png`
  - `semantic_pass_at_1`: pass@1-style proxy (did step 1 exceed a small visibility threshold?)

  These are summarized by [`aggregate_results`](src/scripts/aggregate_results.py:1) into `summary_by_policy.csv`.

  ### Recommended usage patterns (new)

  **A) Cover many starting angles fairly (sweep mode)**

  If you want an “average performance over angles”, set trials to the number of orbital views and sweep:

  ```bash
  python3 -m src.scripts.run_experiments \
    --preset objaverse_subset \
    --policies active active_combined random geometric \
    --trials 24 \
    --initial_view_mode sweep
  ```

  **B) Start from occluded-handle views (semantic_occluded mode)**

  This mode renders all orbital views once per mesh, runs CLIPSeg on the simulator RGB frames, and selects initial views from the lowest-visibility semantic views.
  It’s useful when you explicitly want to test “can the policy recover when the handle starts occluded?”.

  ```bash
  python3 -m src.scripts.run_experiments \
    --preset objaverse_subset \
    --policies active active_combined random geometric \
    --trials 8 \
    --initial_view_mode semantic_occluded \
    --initial_view_occlusion_prompt "handle" \
    --initial_view_occlusion_quantile 0.25
  ```

  Re-sample occluded starting views across runs (while staying deterministic within each run):

  ```bash
  python3 -m src.scripts.run_experiments \
    --preset objaverse_subset \
    --policies active active_combined random geometric \
    --trials 8 \
    --initial_view_mode semantic_occluded \
    --initial_view_seed 1
  ```

  **C) Sweep combined-score hyperparameters from CLI**

  If `active_combined` looks too similar to `active`, try sweeping these:

  ```bash
  python3 -m src.scripts.run_experiments \
    --preset objaverse_subset \
    --policies active active_combined \
    --trials 8 \
    --combined_semantic_variance_weight 1.0 \
    --semantic_threshold 0.30 \
    --semantic_gamma 3.0
  ```

  Notes:
  - Higher `--combined_semantic_variance_weight` increases the influence of semantic disagreement across multiview painting.
  - `--semantic_threshold` / `--semantic_gamma` make the score focus more on high-confidence semantic regions.

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
- Real-world mesh assets (e.g., Objaverse) often have inconsistent coordinate frames. The simulator applies a best-effort stable-pose upright normalization by default (see `normalize_stable_pose` in [`SimulatorConfig`](src/config.py:21)).
- See `documents/plan.md` for milestone guidance and `documents/project.md` for the research pitch.

## Troubleshooting

### Network Issues (China)
If you encounter timeouts downloading models:

1.  **Point-E**: Manually download `base_40m_textvec.pt` and `upsample_40m.pt` from OpenAI (or mirrors) and place them in `~/.cache/point_e/`.
2.  **CLIPSeg**: Manually download the model files from `https://hf-mirror.com/CIDAS/clipseg-rd64-refined` and place them in a local folder. Then set `export CLIPSEG_MODEL_PATH=/path/to/folder` before running.

### Rendering Issues
If you see `ImportError: libGL.so.1` or `Library "GLU" not found`, ensure you installed the system dependencies listed in Quickstart.

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
