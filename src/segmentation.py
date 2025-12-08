"""CLIPSeg affordance masking and 2D-to-3D projection utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from .config import SegmentationConfig

logger = logging.getLogger(__name__)


class CLIPSegSegmenter:
    """Wrapper for CLIPSeg segmentation with lazy model loading."""

    def __init__(self, cfg: SegmentationConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.device if torch.cuda.is_available() or cfg.device == "cpu" else "cpu")
        self.model = None
        self.processor = None
        self.available = False
        self._init_model()

    def _init_model(self) -> None:
        try:
            from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

            self.processor = CLIPSegProcessor.from_pretrained(self.cfg.model_name)
            self.model = CLIPSegForImageSegmentation.from_pretrained(self.cfg.model_name).to(self.device)
            self.available = True
            logger.info("Loaded CLIPSeg model on %s", self.device)
        except Exception as exc:  # pragma: no cover - optional dependency
            logger.warning("CLIPSeg unavailable (%s). Semantic masking will return zeros.", exc)
            self.available = False
            self.model = None
            self.processor = None

    def segment_affordance(self, image: np.ndarray, prompt: Optional[str] = None) -> np.ndarray:
        """Return a float mask in [0,1] with same HxW as input image."""
        if not self.available or self.model is None or self.processor is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        prompt = prompt or self.cfg.prompt
        inputs = self.processor(
            text=[prompt],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # [B, 1, H, W]
            probs = torch.sigmoid(logits)[0, 0]
        mask = probs.detach().cpu().numpy().astype(np.float32)
        return mask


def project_points_to_pixels(
    points_world: np.ndarray,
    cam_to_world: np.ndarray,
    intrinsics,
) -> np.ndarray:
    """Project 3D points (N,3) in world frame to pixel coords (N,2) using intrinsics and pose."""
    world_to_cam = np.linalg.inv(cam_to_world)
    homog = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (world_to_cam @ homog.T).T[:, :3]
    z = cam[:, 2:3]
    z_safe = np.clip(z, 1e-6, None)
    u = intrinsics.fx * cam[:, 0] / z_safe[:, 0] + intrinsics.cx
    v = intrinsics.fy * cam[:, 1] / z_safe[:, 0] + intrinsics.cy
    return np.stack([u, v, cam[:, 2]], axis=1)  # (u, v, depth)


def compute_point_mask_weights(
    points_world: np.ndarray,
    cam_to_world: np.ndarray,
    intrinsics,
    mask: np.ndarray,
) -> np.ndarray:
    """Lookup mask probability for each 3D point by projecting into the image."""
    pix = project_points_to_pixels(points_world, cam_to_world, intrinsics)
    u = pix[:, 0].round().astype(int)
    v = pix[:, 1].round().astype(int)
    depth = pix[:, 2]
    h, w = mask.shape
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depth > 0)
    weights = np.zeros(points_world.shape[0], dtype=np.float32)
    weights[valid] = mask[v[valid], u[valid]]
    return weights
