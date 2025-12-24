"""Point-E integration with multi-seed sampling and auto-alignment."""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from PIL import Image

from .config import PointEConfig
from .pointe_alignment import AlignmentParams, align_clouds_to_reference_front

logger = logging.getLogger(__name__)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class PointEGenerator:
    """Wrapper for Point-E with auto-cropping and multi-seed sampling."""

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
            
            # Load Base
            base_model = model_from_config(MODEL_CONFIGS[base_name], self.device)
            base_model.load_state_dict(load_checkpoint(base_name, self.device))
            base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

            # Load Upsampler
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
        except Exception as exc:  # pragma: no cover
            logger.warning("Point-E unavailable (%s). Using synthetic fallback.", exc)
            self.available = False
            self.sampler = None

    def _get_crop_transform(self, image: np.ndarray, padding: int = 20) -> Tuple[np.ndarray, float, Tuple[float, float]]:
        """Calculate crop centered on object and alignment parameters."""
        h, w = image.shape[:2]
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) if image.ndim == 3 else image
        coords = cv2.findNonZero(gray)
        
        if coords is None:
            return image, 1.0, (0.0, 0.0)

        x, y, bw, bh = cv2.boundingRect(coords)
        
        # Square crop logic
        cx, cy = x + bw / 2, y + bh / 2
        side = max(bw, bh) + padding * 2
        x1 = int(max(0, cx - side / 2))
        y1 = int(max(0, cy - side / 2))
        x2 = int(min(w, x1 + side))
        y2 = int(min(h, y1 + side))
        
        crop_w, crop_h = x2 - x1, y2 - y1
        crop_img = image[y1:y2, x1:x2]
        
        scale = max(crop_w, crop_h) / max(w, h)
        
        # Offset from image center to crop center (normalized)
        full_cx, full_cy = w / 2, h / 2
        crop_cx, crop_cy = x1 + crop_w / 2, y1 + crop_h / 2
        
        off_x = (crop_cx - full_cx) / w
        off_y = (crop_cy - full_cy) / h
        
        return crop_img, scale, (off_x, off_y)

    def _sample_once(self, image: np.ndarray, prompt: str, seed: int) -> np.ndarray:
        _set_seed(seed)
        if not self.available or self.sampler is None:
            return self._synthetic_point_cloud(seed)

        pil_image = Image.fromarray(image.astype(np.uint8)).convert("RGB")
        pil_image = pil_image.resize((256, 256))

        kwargs = dict(texts=[prompt], images=[pil_image])
        samples = None
        try:
            for s in self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=kwargs):
                samples = s
        except Exception as exc:
            logger.warning("Image conditioning failed (%s), retrying text-only.", exc)
            kwargs = dict(texts=[prompt])
            for s in self.sampler.sample_batch_progressive(batch_size=1, model_kwargs=kwargs):
                samples = s

        if samples is None:
            return self._synthetic_point_cloud(seed)

        pc = self.sampler.output_to_point_clouds(samples)[0]
        pts = pc.coords if hasattr(pc, "coords") else pc
        if hasattr(pts, "detach"):
            pts = pts.detach().cpu().numpy()
        
        return np.asarray(pts, dtype=np.float32)

    def _synthetic_point_cloud(self, seed: int) -> np.ndarray:
        _set_seed(seed)
        num = self.cfg.num_points
        theta = np.random.rand(num) * 2 * np.pi
        z = np.random.rand(num) * 0.2 + 0.05
        r = 0.1 + 0.02 * np.random.randn(num)
        return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1).astype(np.float32)

    def generate_point_clouds_from_image(
        self,
        image: np.ndarray,
        prompt: Optional[str] = None,
        num_seeds: Optional[int] = None,
        seed_list: Optional[List[int]] = None,
    ) -> List[np.ndarray]:
        """Generate point clouds with auto-cropping and robust multi-seed alignment.

        Notes:
        - Cropping helps Point-E focus on small objects in a large viewport.
        - Alignment is performed *across seeds* so that global pose jitter does not
          dominate the downstream variance computation.
        """
        prompt = prompt or self.cfg.prompt
        num = num_seeds or self.cfg.num_seeds
        seeds = seed_list or self.cfg.seed_list or list(range(num))
        
        crop_img, _, _ = self._get_crop_transform(image)
        
        clouds: List[np.ndarray] = []
        for i in range(num):
            seed = seeds[i] if i < len(seeds) else seeds[-1] + i + 1
            
            raw_pts = self._sample_once(crop_img, prompt, seed)
            clouds.append(np.asarray(raw_pts, dtype=np.float32))
            
        if self.cfg.align_across_seeds and len(clouds) > 1:
            params = AlignmentParams(
                max_yaw_deg=self.cfg.alignment_max_yaw_deg,
                yaw_step_deg=self.cfg.alignment_yaw_step_deg,
                trim_ratio=self.cfg.alignment_trim_ratio,
                core_ratio=self.cfg.alignment_core_ratio,
                surface_grid=self.cfg.alignment_surface_grid,
                max_points=self.cfg.alignment_max_points,
            )
            try:
                clouds = align_clouds_to_reference_front(clouds, params=params, reference_idx=0, rng_seed=0)
            except Exception as exc:
                logger.warning("Point-E seed alignment failed (%s); returning unaligned clouds.", exc)
        return clouds
