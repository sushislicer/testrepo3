What happened in this run is internally consistent with the code + your debug images: the “active” policy is chasing a 3D centroid
  computed from a broken/uninformative semantic signal, not “keeping the handle in view”.
 
  ### 1) The policy really did move away from the handle (and why)
 
  Your new trajectory.json shows you started at a handle-visible view and then immediately moved to a handle-hidden view:

  - Step 0: view_id = 4 (handle visible) → next_view_id = 0
  - Step 1: view_id = 0 (handle not visible) → next_view_id = 1
  - Step 2: view_id = 1 → next_view_id = 23

  This is exactly what select_next_view_active() does: it picks the next camera whose optical axis best aligns with a single 3D
  “target centroid” (src/experiments.py:159, src/experiments.py:183, src/nbv_policy.py:21). It does not evaluate candidate views by
  “expected handle visibility”.

  Concretely, at step 0 your extracted centroid was:

  - [-0.449, 0.0089, -0.00074] (near the -X boundary of the voxel cube)

  Given the orbital camera layout, a camera at view 0 looks roughly along -X, so it aligns extremely well with a target at negative
  X. I recomputed the active-policy scores for step 0’s centroid: view 0 is the top candidate (score ≈ 0.989), then 23 and 1
  (neighbors). So moving 4 -> 0 is the intended outcome under this policy, even if it loses the handle.

  ### 2) Why the centroid is not “the handle”

  That centroid comes from a pipeline:

  1. Generate multi-seed point clouds from the RGB (Point‑E).
  2. Compute occupancy variance across seeds (variance_grid).
  3. Compute semantic weights by rendering the point cloud into an image, running CLIPSeg “handle”, and projecting mask scores back
     to 3D points (semantic_grid).
  4. Multiply variance_grid * semantic_grid, then min–max normalize per-step, then take top‑k voxels and average them to get the
     centroid (src/experiments.py:156, src/variance_field.py:98, src/variance_field.py:113).

  In your debug artifacts, step 0 clearly shows the semantic part is basically not contributing:

  - The saved semantic “render” images (step_00_cloud0_view0_render.png, etc.) are starfields (sparse dots), not a mug silhouette.
  - The saved CLIPSeg masks (step_00_cloud0_view0_mask.png, etc.) are almost black: in your run the mask max is only ~16/255 ≈ 0.063
    and mean ~0.036 (so CLIPSeg is not confident about “handle” anywhere on that render).

If semantic_grid is near-zero everywhere (or essentially noise), then the “combined score” becomes “variance times tiny numbers”,
  and the min–max normalization can amplify arbitrary outliers. Then extract_topk_centroid() (top 5% of ~885k voxels!) will happily
  produce a centroid dominated by wherever variance/outlier structure exists in the Point‑E clouds—not the true handle location.

  This matches your fig_ghosting_step0.png: the multi-seed overlay has lots of stray clusters/outliers, and the variance field looks
  dominated by broad boundary structure rather than a localized “handle” blob.

  ### 3) The root cause behind the starfield renders (this is the big one)

  Those *_cloud0_view0_render.png images are produced by render_point_cloud_view() (src/segmentation.py:208) which projects 3D points
  using:

  - cam_to_world = pose_to_matrix(poses[current]) from the mesh simulator camera
  - project_points_to_pixels(points_world, cam_to_world, intrinsics) which assumes points_world are in the same world frame as that
    camera (src/segmentation.py:157)

  But the Point‑E point clouds are not guaranteed to be in the simulator’s world frame/scale/orientation. In this code, they are
  passed directly as points_world without any estimated extrinsics/scale alignment to the simulator camera. If the Point‑E cloud is
  scaled differently, shifted, rotated, or effectively in a canonical/object frame, then projecting it with the simulator camera will
  put most points out of the frustum / behind the camera / extremely depth-distorted → exactly the sparse “random dots” you’re
  seeing.

  That breaks “semantic painting” at the source: CLIPSeg is being asked to segment “handle” from a point-splat starfield, so it
  outputs near-zero everywhere, so the policy target centroid stops meaning “handle”.

  ### 4) Why increasing guidance_scale didn’t help

  You increased guidance_scale to 12.1 (confirmed in the saved config), but the bottleneck here isn’t “Point‑E refused to imagine a
  handle” as much as:

  - the semantic channel is effectively dead because its input render is not a coherent 2D depiction of the object, and/or
  - the policy is not optimizing for handle visibility anyway; it optimizes alignment to whatever centroid falls out of the noisy
    score volume.

  ### 5) One more conceptual issue: no belief accumulation

  Each step recomputes Point‑E + variance from scratch from the new view; there’s no “global posterior” update that guarantees
  uncertainty should go down. So it’s completely possible (and in your logs it happens) that variance_sum decreases then increases
  again—this isn’t a bug so much as “not actually doing multi-view fusion”.

  If you want, the fastest way to confirm the primary failure mode (frame/scale mismatch) without changing code is: render the same
  Point‑E cloud with a camera fitted to the cloud bounds (or just visualize it with Open3D) and compare against
  *_cloud0_view0_render.png. If it suddenly looks like a mug under a fitted camera but not under the simulator camera, that pins the
  issue squarely on the projection alignment used in src/segmentation.py:208.