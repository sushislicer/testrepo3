# Project Proposal: Active-Hallucination
## Leveraging Generative Variance to Guide Next-Best-View Selection for Semantic Grasping

**Course Level:** Level-3 (Novel Design & Extensive Experiments)
**Type:** Pipeline Integration / Novel Application of Generative Models
**Core Innovation:** Turning the "hallucination" weakness of generative AI into a "uncertainty" sensor signal for active view selection and (simulated) robotic control.

---

## 1. Abstract

Robotic manipulation in unstructured environments remains a challenge, particularly when critical affordance regions (e.g., the handle of a mug or the trigger of a drill) are occluded from the robot's initial viewpoint. Current state-of-the-art methods, such as *Grasp-Anything-6D* or *CoPa*, typically rely on single-shot reconstruction from depth sensors or exhaustive scanning policies, which fail when the object's geometry is ambiguous or when depth sensors are unavailable.

We propose **Active-Hallucination**, a novel pipeline that utilizes the stochastic nature of text-to-3D generative models (Point-E) as a signal for active perception. We observe that when generative models attempt to reconstruct occluded geometry, they exhibit high variance across different random seeds (hallucination), whereas visible geometry remains consistent. By generating multiple 3D hypotheses from a single RGB view and computing a **Semantic Variance Field**, our system identifies specific regions of high uncertainty regarding the target affordance. We introduce a lightweight Next-Best-View (NBV) heuristic that directs a virtual camera in simulation toward the "center of hallucination." Unlike traditional NBV algorithms that optimize for geometric coverage, our method optimizes for **semantic confidence**, allowing an agent to proactively investigate and "grasp" objects in a simulated environment based on generative priors rather than blind exploration. We evaluate this pipeline purely in simulation on tabletop objects, comparing against random and geometric NBV baselines using variance reduction and handle-localization metrics.

---

## 2. Motivation & The "Novelty Gap"

To satisfy the Level-3 grading criteria, this project must distinguish itself from existing literature. We address specific gaps found in recent papers:

### 2.1. The "Smooth Front" Problem (vs. Traditional NBV)
Traditional Next-Best-View (NBV) algorithms operate on **Geometric Entropy**. They look for "holes" in the 3D map and move the camera to fill them.
*   **The Failure Case:** Consider a robot looking at a pitcher from the front. It sees a smooth, complete cylinder. The handle is hidden entirely behind it. A traditional NBV algorithm sees a "complete" surface with no holes and terminates, assuming the map is done. It misses the handle entirely.
*   **Our Solution:** Our system feeds the image to a Generative Model. The model knows (from training on billions of images) that pitchers usually have handles.
    *   *Seed 1:* Generates a handle on the left.
    *   *Seed 2:* Generates a handle on the right.
    *   *Seed 3:* Generates no handle.
    *   **Result:** Our system detects high variance behind the object. Even though the robot cannot *see* a hole, the AI suspects one should be there. This **Generative Prior** drives the robot to check the back, finding the handle.

### 2.2. The "Sensor Dependency" Problem (vs. Grasp-Anything-6D)
*   **The Baseline:** *Grasp-Anything-6D* (arXiv 2024) relies on a dataset of 1 million ground-truth point clouds and assumes the input is a valid, high-quality point cloud from a depth sensor.
*   **Our Novelty:** We assume **Zero-Shot / Sensor-Less** operation. We do not require a depth sensor. We rely on the generative model to "hallucinate" the 3D structure from a standard RGB web image. Our novelty is dealing with the *noise* of this hallucination, whereas *Grasp-Anything-6D* assumes the input data is valid.

### 2.3. The "Static Planning" Problem (vs. CoPa / VoxPoser)
*   **The Baseline:** *CoPa* (IROS 2024) uses VLMs to plan constraints ("grasp the handle"). However, it assumes the perception module can *find* the handle. If the handle is occluded, the planner fails.
*   **Our Novelty:** We introduce an **Active Loop**. We do not just plan *how* to grasp; we plan *where to look* to make the grasp feasible.

---

## 3. Methodology: The Technical Pipeline

This project is **Inference-Only** (no training required), minimizing "Implementation Hell," and is evaluated **purely in simulation** (no physical robot or sensor hardware required).

### Module A: Stochastic 3D Generation (The "Scatter")
**Input:** A single RGB crop of the object $I_{rgb}$ and a text prompt (e.g., "A blue ceramic pitcher").
**Process:**
We utilize OpenAI’s **Point-E** (image-to-point-cloud diffusion model). We run the inference loop $K$ times (e.g., $K=5$) with different random seeds.
**Output:** A set of $K$ point clouds $\{P_1, P_2, ..., P_k\}$.
*   *Observation:* Visible parts (the front of the pitcher) will align almost perfectly across all $P$. Occluded parts (the handle) will vary wildly in position and shape, or appear as floating noise.

### Module B: The Semantic Variance Field (The Math Contribution)
We must convert these $K$ point clouds into a single metric of uncertainty.

1.  **Voxelization:** We define a shared 3D voxel grid $V$ encompassing the object. For each point cloud $P_i$, we mark voxels as occupied (1) or empty (0). This gives us $K$ occupancy grids.
2.  **Variance Calculation:** For each voxel $v(x,y,z)$, we calculate the variance of occupancy across the $K$ grids:
    $$ \sigma^2(v) = \frac{1}{K} \sum_{i=1}^{K} (O_i(v) - \bar{O}(v))^2 $$
    *   $\sigma^2 \approx 0$: High confidence (either definitely empty or definitely occupied).
    *   $\sigma^2 > \text{Threshold}$: High Hallucination (the model is guessing).

### Module C: Semantic Masking (The "Intent")
High variance isn't enough—the model might hallucinate the floor or background. We only care about the **Affordance** (e.g., the Handle).

1.  **2D Segmentation:** We run **CLIPSeg** or **GroundingDINO** on the input image $I_{rgb}$ with the prompt "Handle". This gives us a 2D mask $M$.
2.  **3D Projection:** We project this mask into the voxel grid.
3.  **Targeted Uncertainty:** We filter the Variance Field to only consider voxels that are semantically relevant (or neighbors of relevant voxels).
    $$ \text{Score}(v) = \sigma^2(v) \times \text{SemanticWeight}(v) $$

### Module D: Active View Selection (The Control, in Simulation)
1.  **Centroid Extraction:** We calculate the 3D centroid $C_{target}$ of the highest-scoring voxels (the "Center of Hallucination").
2.  **Vector Calculation:** We compute a vector from the current camera position to $C_{target}$.
3.  **Action:** A virtual camera in a simulated tabletop scene moves along an orbital path by $\theta$ degrees (e.g., $30^{\circ}$) to align its optical axis with $C_{target}$.
4.  **Loop:** Render a new view. If variance drops below a threshold $\tau$, we mark the handle as sufficiently observed and trigger a (simulated) grasp success; otherwise, we repeat.

### Simulation Platform: Python Virtual Camera
We implement the simulated tabletop environment and camera entirely in Python using lightweight 3D libraries (e.g., **trimesh** + **pyrender**, optionally **Open3D** for debugging). These tools allow us to: (1) load mesh models of objects, (2) place them on a virtual table, (3) define camera intrinsics and extrinsics, and (4) render RGB images offscreen from arbitrary viewpoints. Our NBV module outputs target camera poses on a discrete orbital ring; the Python renderer then produces the corresponding "robot view" image that is fed back into Point-E and CLIPSeg. No external robotics simulator or GUI is required—everything runs as a pure Python codebase.

---

## 4. Implementation Plan (4 Weeks)

### Week 1: The "Scatter" (Generation Scripting)
*   **Goal:** Generate 5 aligned point clouds from 1 image.
*   **Tools:** Python, PyTorch, Point-E (HuggingFace), trimesh, pyrender, Open3D.
*   **Task:** Write a script that loops the model 5 times. Use Open3D (or pyrender) to visualize them superimposed. You should see the "fuzzy ghost" effect on hidden parts.

### Week 2: The "Gather" (Metric Implementation)
*   **Goal:** Calculate the Variance Cloud.
*   **Task:**
    1.  Implement a simple Voxel Grid in numpy.
    2.  Map points to voxels.
    3.  Calculate variance per voxel.
    4.  Visualize the result: A point cloud where **Color = Variance** (e.g., Red = High Variance, Blue = Low).

### Week 3: Semantic Layer & Logic
*   **Goal:** Integrate CLIPSeg and the Control Logic.
*   **Task:**
    1.  Run CLIPSeg on the input image.
    2.  Mask the variance cloud.
    3.  Compute the centroid of the remaining "red" points.
    4.  Output a text command: "Recommended Action: Move Camera [Left/Right/Up] by X degrees."

### Week 4: Simulation Experiments & Report
*   **Goal:** Run Active-Hallucination in a controlled simulated tabletop setup.
*   **Task:** Select 5–10 mesh models of household objects (e.g., mugs, pitchers, drills). For each object, define a discrete set of viewpoints on an orbital camera ring in the Python simulator, render "bad" initial views plus follow-up views according to our NBV policy and baselines (Random, Geometric NBV) using the pyrender-based virtual camera. Run the full pipeline, record camera steps, variance trajectories, and handle-localization success, and prepare figures for the report and presentation.

---

## 5. Evaluation Plan

To secure a Level-3 grade, you need rigorous evaluation. Since physical robot trials are slow, use **Simulated Verification** on a dataset like **Objaverse** or **YCB**.

### 5.1. Quantitative Metrics

**Baseline Comparison:**
Compare **Active-Hallucination** against two baselines:
1.  **Random Policy:** Move camera to a random adjacent view.
2.  **Geometric NBV (Farthest Point):** Move to the view furthest from the current one (optimizing for coverage).

**Metric 1: Variance Reduction Rate (VRR)**
*   Measure the total sum of variance in the voxel grid after Step 1, Step 2, and Step 3.
*   **Hypothesis:** Your method will reduce variance faster (steepest slope) because it targets the uncertainty directly.

**Metric 2: Semantic Grasp Success @ Step K**
*   After $K$ moves, generate the final mesh. Attempt to locate the handle centroid.
*   **Success:** Distance between *Predicted Handle Centroid* and *Ground Truth Handle Centroid* $< 2.0$cm.
*   **Hypothesis:** Your method reaches success at Step 1 or 2. Baselines may take 3+ steps or fail.

### 5.2. Qualitative Visuals (The "Cool Factor")

**Visualization 1: The "Ghosting" Effect**
Create a figure with 3 columns:
1.  **Input:** Image of a mug with the handle hidden.
2.  **Raw Output:** 5 Point-E clouds overlaid. Show the handle appearing as a blurry, scattered mess (the "ghost").
3.  **Variance Map:** A voxel grid where the handle region is glowing Red (High Uncertainty).

**Visualization 2: The Trajectory**
Show a sequence of 3 images:
1.  **Step 0:** Robot sees front of pitcher (Variance High on back). Action: "Move Left."
2.  **Step 1:** Robot sees side of pitcher (Handle partially visible). Variance decreases.
3.  **Step 2:** Robot sees handle clearly. Variance nears zero. **Grasp Executed.**

---

## 6. Why This Fits the Class Criteria

*   **Novel Design:** It proposes a new metric (**Semantic Generative Variance**) that does not exist in standard robotics literature. It combines Generative AI with Active Perception in a way *Grasp-Anything-6D* does not.
*   **Extensive Experiments:** The evaluation plan covers both quantitative analysis (VRR, Success Rate) and qualitative proofs (visualizations) on multiple objects.
*   **Feasibility:** It avoids "Implementation Hell" by being **Inference-Only**. You are not training a diffusion model (which takes months); you are just running it and doing math on the output.
*   **Pipeline Integration:** It perfectly demonstrates the "gluing strong pre-trained models together" approach (Point-E + CLIPSeg + Custom Logic).
