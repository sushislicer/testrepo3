"""
Active-Hallucination package

Modules:
    config: configuration dataclasses and YAML helpers.
    simulator: pyrender-based tabletop simulator and camera utilities.
    pointe_wrapper: Point-E integration with multi-seed sampling.
    variance_field: voxel grid and semantic variance computation.
    segmentation: CLIPSeg affordance masking and 3D projection.
    nbv_policy: next-best-view strategies (active, random, geometric).
    experiments: orchestration of simulation loops and metrics.
    visualization: plotting utilities for figures and diagnostics.
"""

from .config import (
    SimulatorConfig,
    CameraIntrinsics,
    PointEConfig,
    VarianceConfig,
    SegmentationConfig,
    PolicyConfig,
    ExperimentConfig,
    ActiveHallucinationConfig,
)

__all__ = [
    "SimulatorConfig",
    "CameraIntrinsics",
    "PointEConfig",
    "VarianceConfig",
    "SegmentationConfig",
    "PolicyConfig",
    "ExperimentConfig",
    "ActiveHallucinationConfig",
]
