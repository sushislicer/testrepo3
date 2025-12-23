# Active-Hallucination – Step-by-Step Implementation Plan

This document turns the high-level proposal in `documents/project.md` into a concrete, executable plan. It is organized as dependency-ordered phases with detailed tasks, suggested outputs, and checks. Adjust dates based on your actual schedule, but try to keep the relative ordering.

---

## 0. Goals, Scope, and Constraints

- **Goal:** Implement and evaluate Active-Hallucination, a simulation-only pipeline that uses generative variance from Point-E plus CLIPSeg-based semantic *painting* (per Point‑E hypothesis via multiview renders) to drive a Next-Best-View (NBV) policy for revealing object affordances (e.g., handles).
- **Scope:** Pure Python codebase; no physical robots or real sensors. Virtual tabletop scenes, orbital camera ring, RGB rendering, and evaluation all run in software.
- **Core components:**
  - Python virtual camera simulator (`trimesh` + `pyrender` / `open3d`).
  - Point-E (image-to-point-cloud) integration with multi-seed sampling.
  - Semantic Variance Field (voxelized uncertainty metric).
  - CLIPSeg-based semantic painting of Point‑E hypotheses via multiview renders.
  - NBV policy over a discrete set of camera poses, plus baselines (Random, Geometric NBV).
  - Experimental harness, metrics, visualizations, and report-ready figures.
- **Key deadlines (from requirements):**
  - Presentation 1: **2025-12-18** (minimum working demo + visuals).
  - Presentation 2: **2025-12-25** (more complete experiments).
  - Final report + code: **2026-01-25 24:00**.

---

## 1. Environment and Repository Setup

**1.1. Python / CUDA environment**

- Decide on environment manager (e.g., `conda` or `venv`).
- Create a fresh environment (Python 3.9+ recommended).
- Install core packages:
  - `torch`, `torchvision`, `torchaudio` (GPU-enabled if possible).
  - `numpy`, `scipy`, `matplotlib`, `tqdm`.
  - `trimesh`, `pyrender`, `open3d` (for mesh handling and visualization).
  - `opencv-python` (for basic image I/O and visualization).
  - HuggingFace ecosystem: `transformers`, `huggingface-hub`, `accelerate`, `safetensors`.
  - Any Point-E-specific package or repo requirements.
  - CLIPSeg dependencies (e.g., `pillow`, `torchvision`, relevant model loader).
- Verify GPU availability in Python (simple `torch.cuda.is_available()` check).

**1.2. Repository structure**

- Introduce a simple, modular source tree (if not already present), for example:
  - `src/active_hallucination/`
    - `__init__.py`
    - `config.py` – Config objects or YAML loader.
    - `simulator.py` – Virtual tabletop + camera rendering.
    - `pointe_wrapper.py` – Point-E loading and multi-seed sampling utilities.
    - `variance_field.py` – Voxel grid + Semantic Variance computation.
    - `segmentation.py` – CLIPSeg integration + multiview 3D painting helpers.
    - `nbv_policy.py` – NBV policy + baselines.
    - `experiments.py` – High-level experiment loops and logging.
    - `visualization.py` – Plotting and figure generation utilities.
  - `scripts/`
    - Small entrypoints, e.g., `run_single_object_demo.py`, `run_experiments.py`.
- Add a minimal `README.md` stub describing how to set up the environment and run a basic test (to be extended later).

**Deliverables / checks:**

- `python -c "import torch, trimesh, pyrender, open3d"` runs without errors.
- Basic repo structure exists and is importable (`python -c "import active_hallucination"` from `src` root if using a package).

---

## 2. Assets: Mesh Models and Scene Definition

**2.1. Object mesh selection**

- Decide on 5–10 object categories relevant to handles/affordances (e.g., mugs, pitchers, kettles, drills).
- For each category, select at least 1 (ideally 2–3) mesh models:
  - Format: `OBJ`, `PLY`, `GLB`, or `STL` supported by `trimesh`/`open3d`.
  - Ensure meshes are reasonably watertight and have sensible scale.
- Store them under, e.g., `assets/meshes/<category>/<object_id>.obj`.

**2.2. Tabletop scene conventions**

- Define a canonical tabletop coordinate frame (e.g., table plane at `z = 0`, object centered at origin).
- Decide on object placement:
  - Translate object so its bottom touches `z=0` and it is centered at `(0,0)`.
  - Optionally randomize small rotations around `z` for data augmentation later.
- Specify a default light setup for pyrender (e.g., directional + ambient lights).

**Deliverables / checks:**

- A small Python script (e.g., `scripts/inspect_meshes.py`) that loads each mesh and visualizes it with `open3d` or `pyrender`, confirming correct orientation and scale.

---

## 3. Python Virtual Camera Simulator

**3.1. Camera parameterization**

- Define camera intrinsics:
  - Focal length, principal point, image resolution (e.g., 512×512).
  - Fixed for all experiments for simplicity.
- Define extrinsics via orbital parameterization:
  - Radius `R` from object center.
  - Elevation angle (fixed or a small set, e.g., 30° above table).
  - Azimuth angle around the object (e.g., discrete steps every 15° → 24 views).
- Represent camera poses either as:
  - 4×4 homogeneous transforms, or
  - `position` + `look_at` + `up` vectors.

**3.2. Implementation in `simulator.py`**

- Implement a function to generate the discrete orbital camera ring:
  - `generate_orbital_poses(num_views, radius, elevation_deg) -> list[Pose]`.
- Implement a function to render a single view:
  - `render_view(mesh_path, camera_pose, intrinsics) -> rgb_image` using `pyrender`.
  - Use offscreen rendering (no GUI) and return a NumPy array / `PIL.Image`.
- Optionally support depth map rendering for debugging (even if not used in method).

**3.3. Simple demo**

- Create `scripts/demo_orbital_camera.py` that:
  - Loads one mesh.
  - Generates the camera ring.
  - Renders multiple views and saves them as images to `outputs/demo_orbit/`.

**Deliverables / checks:**

- A folder of rendered RGB images showing the object from different azimuth angles.
- Code path: `active_hallucination/simulator.py` and `scripts/demo_orbital_camera.py` working end-to-end.

---

## 4. Point-E Integration and Multi-Seed Generation

**4.1. Model loading and configuration**

- Decide which Point-E variant to use (image-to-point-cloud diffusion model).
- Either:
  - Install the official Point-E package, or
  - Clone the repo and add a thin wrapper module `pointe_wrapper.py` that exposes a simple API.
- Implement model initialization code that:
  - Downloads (or loads) pretrained weights.
  - Supports GPU execution when available.

**4.2. Single-view, multi-seed sampling**

- Implement a function in `pointe_wrapper.py`, e.g.:
  - `generate_point_clouds_from_image(image: np.ndarray, prompt: str, num_seeds: int, seed_list: Optional[list[int]]) -> list[np.ndarray]`.
- Under the hood:
  - For each seed, set the random seed (Python, NumPy, and PyTorch as needed).
  - Run Point-E to get a point cloud in a consistent coordinate frame.
  - Convert the output to a standard NumPy array of shape `(N_points, 3)`.

**4.3. Alignment and sanity checks**

- For a fixed input image and prompt, generate `K` point clouds (e.g., `K=5`).
- Visualize them overlaid using `open3d`:
  - Check that the visible front region of the object is relatively stable across clouds.
  - Inspect that occluded regions show variability / “ghosting” behavior.

**Deliverables / checks:**

- `scripts/demo_pointe_multiseed.py` that takes a rendered RGB image and outputs a color-coded overlay of K point clouds for manual inspection.

---

## 5. Semantic Variance Field (Voxel Grid)

**5.1. Voxel grid definition**

- Decide on the voxel grid bounds and resolution:
  - E.g., cube `[-B, B]^3` around object center (B ~ object radius).
  - Grid resolution `N^3` (e.g., 64³ or 128³, depending on speed/memory trade-offs).
- Implement a voxelizer in `variance_field.py`:
  - Map each point `(x, y, z)` to voxel indices `(i, j, k)`.
  - Handle points outside bounds by discarding or clamping.

**5.2. Variance computation**

- Choose the basic statistic per voxel:
  - Option A: Store point count per voxel for each seed → treat occupancy as a binary variable and compute variance over occupancy.
  - Option B: Store mean position per voxel per seed and compute variance of positions (if enough points per voxel).
- Implement functions:
  - `accumulate_point_clouds(point_clouds) -> voxel_stats`.
  - `compute_variance_field(voxel_stats) -> variance_grid` (3D array of variance values).
- Normalize or smooth the variance field if needed (e.g., Gaussian smoothing for stability).

**5.3. Visualization and debugging**

- Implement helper to export high-variance voxels as a point cloud for visualization in `open3d` (color = variance).
- Create a demo script, `scripts/demo_variance_field.py`, that:
  - Loads a test RGB image.
  - Runs multi-seed Point-E.
  - Computes the variance field.
  - Visualizes high-variance voxels vs. low-variance voxels.

**Deliverables / checks:**

- Clear “ghosting” behavior in high-variance voxels for a view with occluded handle.

---

## 6. CLIPSeg Integration and Semantic Painting

**6.1. 2D CLIPSeg pipeline**

- In `segmentation.py`, load a CLIPSeg model (via HuggingFace if applicable).
- Implement:
  - `segment_affordance(image: np.ndarray, text_prompt: str) -> mask: np.ndarray`.
- The mask should be a binary or float array with the same resolution as the input image (e.g., 512×512), indicating probability of the affordance (e.g., “handle”).

**6.2. Multiview painting: render → CLIPSeg → project back to 3D**

- Goal: assign a **3D affordance probability** to each Point‑E generation, even if the affordance is occluded in the input view.
- For each Point‑E generation `P_k`:
  1. Render `P_k` into **3 views** around the object:
     - View 0: match the *input camera* pose (the simulator view that produced the RGB fed to Point‑E).
     - View 1: rotate the camera by +120° yaw around the object center.
     - View 2: rotate the camera by +240° yaw around the object center.
  2. Run CLIPSeg on each render to obtain masks `M_k^0, M_k^1, M_k^2`.
  3. Project the 3D points of `P_k` into each view and sample the mask values to obtain per-point weights `w_k^i`.
  4. Aggregate the per-point weights across views (recommended: `max_i w_k^i`) to get a final semantic weight `w_k` for that generation.
- Finally, aggregate per-point weights into a voxel grid (e.g., average weights per voxel) to obtain `SemanticWeight(v)` for the score volume.

**6.3. Combined semantic variance score**

- Implement the final voxel score as in the proposal:
  - `Score(v) = Variance(v) * SemanticWeight(v)`.
- Normalize scores to a convenient range (e.g., 0–1).

**Deliverables / checks:**

- For a mug with hidden handle, visualize the semantic-weighted variance field and confirm that the region behind the mug (where the handle likely is) lights up strongly.

---

## 7. NBV Policy and Baselines

**7.1. Candidate view set**

- Reuse the orbital camera ring from the simulator (Section 3).
- Index each camera pose by an integer ID for convenience.

**7.2. Active-Hallucination NBV policy**

- In `nbv_policy.py`, implement:
  - `select_next_view(current_view_id, variance_score_grid, camera_ring) -> next_view_id`.
- Strategy:
  - Compute the 3D centroid of the top-k highest scoring voxels (Center of Hallucination).
  - For each candidate view, compute how well its optical axis aligns with that centroid (e.g., cosine similarity).
  - Select the view with the highest alignment score that is not yet visited (or limit to neighbors if you want local moves).

**7.3. Baseline 1 – Random policy**

- Implement:
  - `select_next_view_random(current_view_id, visited_views, camera_ring) -> next_view_id`.
- Strategy: pick a random unvisited neighboring view or any unvisited view on the ring.

**7.4. Baseline 2 – Geometric NBV (Farthest point)**

- Implement:
  - `select_next_view_geometric(current_view_id, visited_views, camera_ring) -> next_view_id`.
- Strategy: choose the view with maximum angular distance from the current view (or from the mean of visited views), ignoring semantic variance.

**Deliverables / checks:**

- Unit tests or small scripts that run each policy on dummy data and confirm behavior (e.g., Active-Hallucination tends to move toward the hidden side in a simple synthetic setup).

---

## 8. Simulation Loop and Metrics

**8.1. Per-object simulation loop**

- In `experiments.py`, implement a function like:
  - `run_simulation_for_object(object_mesh, prompt, num_steps, policy) -> trajectory_stats`.
- For each trial:
  1. Choose an initial “bad” view (e.g., front view where handle is occluded).
  2. For each step `t`:
     - Render RGB from current view.
     - Run Point-E multi-seed generation.
     - For each Point‑E generation: render 3 views (0/120/240° yaw), run CLIPSeg per render, and project masks back to 3D to get semantic weights.
     - Compute variance field and semantic score.
     - Log total variance, semantic statistics, and any debug images.
     - Invoke policy to get `next_view_id`.
     - Move to next view.
  3. After `K` steps, compute final handle localization and mark success/failure.

**8.2. Metric 1 – Variance Reduction Rate (VRR)**

- For each step, compute `total_variance = sum(variance_grid)`.
- Define VRR as the normalized decrease over steps for each method.
- Store per-step values in logs for plotting.

**8.3. Metric 2 – Semantic grasp success @ step K**

- Define ground-truth handle centroid for each mesh:
  - Approximate manually (e.g., by sampling mesh points labeled as “handle”), or by precomputing from a mask in the mesh frame.
- At the final step, compute predicted handle centroid:
  - Use the high-scoring voxels (semantic variance score) to find 3D centroid.
- Define success when distance between predicted and ground-truth centroids `< 2 cm` (or a chosen threshold in mesh units).
- Log success/failure per trial for each method and compute success rates.

**Deliverables / checks:**

- A CSV or JSON log per run containing per-step variance, chosen views, and success labels.

---

## 9. Experimental Design and Execution

**9.1. Object and initial-view selection**

- For each object:
  - Manually identify 1–2 initial views where the target affordance is occluded.
  - Confirm visually that the affordance becomes visible from at least one other view on the ring.

**9.2. Number of trials and seeds**

- For each object and each policy (Active-Hallucination, Random, Geometric NBV):
  - Run multiple trials with different random seeds (for Point-E and Random policy).
  - Decide on `num_trials` (e.g., 5–10 per object-policy combination) balancing time and statistical significance.

**9.3. Ablations**

- Consider simple ablations if time permits:
  - Vary `K` (number of Point-E samples): e.g., 3 vs. 5 vs. 8.
  - Turn off semantic weighting (variance only) to quantify benefit of CLIPSeg.
  - Coarser vs. finer voxel grids.

**Deliverables / checks:**

- A clear experimental config file or script specifying which objects, views, steps, and seeds are used for the “main” experiments and any ablations.

---

## 10. Visualization and Qualitative Results

**10.1. Ghosting effect figure**

- Implement utilities in `visualization.py` to:
  - Overlay K point clouds in different colors / transparencies.
  - Render them from a fixed “viewer” camera to highlight ghosting around occluded regions.
- Generate the 3-column figure described in the proposal:
  1. Input RGB image (handle occluded).
  2. Raw multi-seed Point-E outputs (overlaid point clouds).
  3. Variance map visualization (high-variance voxels highlighted).

**10.2. Trajectory visualization**

- For each method (especially Active-Hallucination):
  - Render the sequence of RGB images along the chosen trajectory (Step 0, 1, 2, …).
  - Overlay or annotate with arrows and variance maps if possible.
- Assemble the 3-step “trajectory” figure:
  1. Step 0: front view, high variance behind object.
  2. Step 1: side view, variance reduced.
  3. Step 2: handle clearly visible, variance low, (simulated) grasp executed.

**10.3. Metric plots**

- Plot VRR curves (variance vs. step) for each method.
- Plot bar charts for success @ step K per method and per object category.

**Deliverables / checks:**

- A set of publication-quality figures saved under `outputs/figures/`, ready to insert into slides and report.

---

## 11. Reproducibility, CLI, and README

**11.1. Command-line entrypoints**

- Create small user-facing scripts (or `if __name__ == "__main__"` blocks) that allow:
  - Running a single-object demo: `python -m active_hallucination.scripts.run_single_object_demo --mesh assets/meshes/mug_01.obj --policy active`.
  - Running the full experiment suite: `python scripts/run_experiments.py --config configs/main.yaml`.

**11.2. Configuration management**

- Use a simple config system (YAML/JSON or a Python dataclass) for parameters like:
  - Camera ring size, K (number of Point-E samples), voxel size, number of steps, etc.

**11.3. README and documentation**

- Extend the project `README.md` to include:
  - Environment setup instructions.
  - How to download any external assets/models (without embedding links if not allowed, but naming sources clearly).
  - Example commands for demos and full experiments.
  - Expected outputs and how to interpret them.

**Deliverables / checks:**

- A TA should be able to read the README, set up the environment, and reproduce at least one of your main results using one or two commands.

---

## 12. Presentation and Report Preparation

**12.1. Slides for Dec 18 / Dec 25**

- For the first presentation (Dec 18):
  - Emphasize motivation, method overview, and early qualitative demos (ghosting + simple trajectory).
  - Include preliminary VRR curves if available for at least one object.
- For the second presentation (Dec 25):
  - Add more complete quantitative results and ablations.
  - Show polished figures from Section 10.

**12.2. Final report (due 2026-01-25)**

- Use the NeurIPS LaTeX template as required.
- Suggested structure:
  - Introduction + motivation.
  - Related work (NBV, generative models, grasping).
  - Method (Point-E + Semantic Variance Field + CLIPSeg + NBV policy).
  - Experimental setup (simulator details, objects, metrics, baselines).
  - Results & discussion (quantitative + qualitative).
  - Limitations and future work (e.g., extension to real robots).
  - Contribution summaries for each group member.

**12.3. Final code cleanup**

- Remove dead code and debug scripts that are not needed.
- Ensure all paths are relative and no hard-coded absolute paths remain.
- Tag or branch the repo as the “final submission” version to keep it stable.

**Deliverables / checks:**

- Slide decks for both presentations saved under `docs/slides/`.
- Final report LaTeX project and compiled PDF in `docs/report/`.

---

This plan is intentionally detailed; you don’t need to implement every optional ablation or visualization to pass the course. Prioritize getting a robust end-to-end pipeline (Sections 1–8), then focus on the most impactful experiments and figures (Sections 9–10), and finally polish reproducibility and writing (Sections 11–12).
