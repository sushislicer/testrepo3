"""Point-E integration with multi-seed sampling and graceful CPU fallback."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .config import PointEConfig

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PointEGenerator:
    """Thin wrapper around OpenAI's Point-E for multi-seed point cloud sampling.

    When the Point-E package or weights are unavailable, it falls back to
    generating structured random point clouds to keep the pipeline runnable
    for debugging on CPU-only machines.
    """

    def __init__(self, cfg: PointEConfig):
        self.cfg = cfg
        self.available = False
        self.sampler = None
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        self._init_model()

    def _init_model(self) -> None:
        try:
            from point_e.diffusion.sampler import PointCloudSampler
            from point_e.models.configs import MODEL_CONFIGS, model_from_config
            from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
            from point_e.models.download import load_checkpoint

            base_name = "base40M"
            upsampler_name = "upsample"
            base_model = model_from_config(MODEL_CONFIGS[base_name], self.device)
            base_model.load_state_dict(load_checkpoint(base_name, self.device))
            base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

            up_model = model_from_config(MODEL_CONFIGS[upsampler_name], self.device)
            up_model.load_state_dict(load_checkpoint(upsampler_name, self.device))
            up_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[upsampler_name])

            self.sampler = PointCloudSampler(
                device=self.device,
                models=[base_model, up_model],
                diffusions=[base_diffusion, up_diffusion],
                num_points=[1024, max(self.cfg.num_points - 1024, 0)],
                aux_channels=["R", "G", "B"],
                guidance_scale=[self.cfg.guidance_scale, self.cfg.guidance_scale],
            )
            self.available = True
            logger.info("Loaded Point-E models successfully on %s", self.device)
        except Exception as exc:  # pragma: no cover - best effort init
            logger.warning("Point-E unavailable (%s). Falling back to synthetic point clouds.", exc)
            self.available = False
            self.sampler = None

    def _sample_once(self, image: np.ndarray, prompt: str, seed: int) -> np.ndarray:
        _set_seed(seed)
        if not self.available or self.sampler is None:
            return self._synthetic_point_cloud(seed)

        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover
            logger.warning("Pillow missing; returning synthetic cloud (%s)", exc)
            return self._synthetic_point_cloud(seed)

        pil_image = Image.fromarray(image.astype(np.uint8))
        pil_image = pil_image.resize((256, 256))
        image_tensor = torch.tensor(np.array(pil_image)).float().to(self.device) / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # 1x3xHxW

        samples = None
        kwargs = dict(texts=[prompt])
        # Try image conditioning if supported by the sampler
        try:
            kwargs["images"] = image_tensor
        except Exception:
            pass

        for sample in self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=kwargs):
            samples = sample

        pcs = self.sampler.output_to_point_clouds(samples)
        pts = pcs[0].coords.detach().cpu().numpy()
        return pts

    def _synthetic_point_cloud(self, seed: int) -> np.ndarray:
        """Generate a structured synthetic cloud (filled cylinder) for debugging."""
        _set_seed(seed)
        num = self.cfg.num_points
        theta = np.random.rand(num) * 2 * np.pi
        z = np.random.rand(num) * 0.2 + 0.05
        r = 0.1 + 0.02 * np.random.randn(num)
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        return np.stack([x, y, z], axis=1).astype(np.float32)

    def generate_point_clouds_from_image(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
        num_seeds: Optional[int] = None,
        seed_list: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """Generate a list of point clouds conditioned on an RGB image."""
        prompt = prompt or self.cfg.prompt
        num = num_seeds or self.cfg.num_seeds
        seeds = seed_list or self.cfg.seed_list or list(range(num))
        clouds: List[np.ndarray] = []
        for i in range(num):
            seed = seeds[i] if i < len(seeds) else seeds[-1] + i + 1
            clouds.append(self._sample_once(image, prompt, seed))
        return clouds
