# Active-Hallucination

Simulation-only pipeline that leverages generative variance (Point-E) plus CLIPSeg semantic masking to drive Next-Best-View selection for revealing object affordances (e.g., handles) in tabletop scenes.

## Quickstart

1) Create environment (example):
```
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
2) Ensure GPU-enabled PyTorch if available (`python - <<'PY'\nimport torch; print(torch.cuda.is_available())\nPY`).
3) Place meshes under `assets/meshes/` (OBJ/PLY/GLB/STL). The code assumes objects are centered near the origin with the base on `z=0`.
4) Run a dry demo (no heavy compute) to verify imports:
```
python -m active_hallucination.scripts.demo_orbital_camera --mesh assets/meshes/example.obj
```

## Layout

- `src/active_hallucination/`: core library
  - `config.py`: dataclasses + YAML helpers.
  - `simulator.py`: pyrender-based tabletop camera + renderer.
  - `pointe_wrapper.py`: Point-E multi-seed generation helper with graceful fallbacks.
  - `variance_field.py`: voxel grid + variance/semantic score computation.
  - `segmentation.py`: CLIPSeg affordance masking.
  - `nbv_policy.py`: Active-Hallucination NBV + baselines.
  - `experiments.py`: end-to-end loops, metrics, logging.
  - `visualization.py`: overlays, variance heatmaps, trajectory grids.
- `scripts/`: CLI entrypoints for demos and experiments.
- `assets/meshes/`: place your mesh models here (see `plan.md` for categories).
- `outputs/`: renders, logs, and figures.
- `configs/`: YAML configs for experiments.

## Notes

- Point-E and CLIPSeg weights are downloaded automatically via HuggingFace when first used. Expect GPU to speed up Point-E; CPU will be slow.
- The pipeline is inference-only; no training loops are included.
- See `documents/plan.md` for milestone guidance and `documents/project.md` for the research pitch.
