"""Configuration dataclasses and YAML helpers for Active-Hallucination."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional
import yaml


@dataclass
class CameraIntrinsics:
    width: int = 512
    height: int = 512
    fx: float = 575.0
    fy: float = 575.0
    cx: float = 256.0
    cy: float = 256.0


@dataclass
class SimulatorConfig:
    radius: float = 1.2  # meters
    elevation_deg: float = 30.0
    num_views: int = 24
    mesh_path: str = "assets/meshes/example.obj"
    light_intensity: float = 3.5
    background_color: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    intrinsics: CameraIntrinsics = field(default_factory=CameraIntrinsics)


@dataclass
class PointEConfig:
    device: str = "cuda"  # fall back to cpu if unavailable
    num_points: int = 4096
    num_seeds: int = 5
    guidance_scale: float = 3.0
    model_name: str = "point-e"  # descriptive; wrapper resolves to actual weights
    prompt: str = "a photo of a mug"
    seed_list: Optional[List[int]] = None


@dataclass
class VarianceConfig:
    grid_bounds: float = 0.5  # cube half-extent in meters
    resolution: int = 96
    min_points_per_voxel: int = 1
    gaussian_sigma: Optional[float] = None
    topk_ratio: float = 0.05  # fraction of voxels used for centroid


@dataclass
class SegmentationConfig:
    prompt: str = "handle"
    threshold: float = 0.25
    model_name: str = "CIDAS/clipseg-rd64-refined"
    device: str = "cuda"


@dataclass
class PolicyConfig:
    alignment_pow: float = 1.5
    min_angle_deg: float = 5.0
    random_seed: int = 42


@dataclass
class ExperimentConfig:
    num_steps: int = 3
    save_debug: bool = True
    output_dir: str = "outputs/logs"
    trajectory_name: str = "demo"
    policy: str = "active"  # active | random | geometric


@dataclass
class ActiveHallucinationConfig:
    simulator: SimulatorConfig = field(default_factory=SimulatorConfig)
    pointe: PointEConfig = field(default_factory=PointEConfig)
    variance: VarianceConfig = field(default_factory=VarianceConfig)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    policy: PolicyConfig = field(default_factory=PolicyConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)

    @staticmethod
    def from_yaml(path: str) -> "ActiveHallucinationConfig":
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return ActiveHallucinationConfig.from_dict(data or {})

    @staticmethod
    def from_dict(data: dict) -> "ActiveHallucinationConfig":
        def _merge(default_obj, override):
            if override is None:
                return default_obj
            merged = asdict(default_obj)
            merged.update(override)
            return merged

        sim = SimulatorConfig(**_merge(SimulatorConfig(), data.get("simulator")))
        pointe = PointEConfig(**_merge(PointEConfig(), data.get("pointe")))
        var = VarianceConfig(**_merge(VarianceConfig(), data.get("variance")))
        seg = SegmentationConfig(**_merge(SegmentationConfig(), data.get("segmentation")))
        pol = PolicyConfig(**_merge(PolicyConfig(), data.get("policy")))
        exp = ExperimentConfig(**_merge(ExperimentConfig(), data.get("experiment")))
        sim.intrinsics = CameraIntrinsics(
            **_merge(CameraIntrinsics(), data.get("simulator", {}).get("intrinsics", {}))
        )
        return ActiveHallucinationConfig(simulator=sim, pointe=pointe, variance=var, segmentation=seg, policy=pol, experiment=exp)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_yaml(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(self.to_dict(), f)
