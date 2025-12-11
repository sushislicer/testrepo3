"""CLIPSeg affordance masking and 2D-to-3D projection utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import cv2
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
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits  # expected [B, 1, H, W]
            probs = torch.sigmoid(logits)
            expected_hw = logits.shape[-2:] if logits.dim() >= 2 else None

        # Reduce to a single 2D mask robustly across possible tensor shapes.
        if probs.dim() == 4:           # [B,1,H,W] or [B,C,H,W]
            probs = probs[0, 0]
        elif probs.dim() == 3:         # [B,H,W] or [C,H,W]
            probs = probs[0]
        elif probs.dim() == 2:         # [H,W] already fine
            pass
        else:                          # anything else is unexpected
            logger.warning("Segmentation mask logits have unexpected dims %s; returning zeros.", list(probs.shape))
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        mask = probs.detach().cpu().numpy().astype(np.float32)

        # Normalize mask shape to 2D and match the rendered image resolution.
        mask = np.squeeze(mask)
        if mask.ndim == 1 and expected_hw is not None:
            h_exp, w_exp = int(expected_hw[0]), int(expected_hw[1])
            if h_exp * w_exp == mask.size:
                mask = mask.reshape(h_exp, w_exp)
        if mask.ndim == 1:
            # If still flat, try square reshape as last resort.
            side = int(np.sqrt(mask.size))
            if side * side == mask.size:
                mask = mask.reshape(side, side)
        if mask.ndim == 3:
            mask = mask.mean(axis=-1)
        if mask.ndim != 2:
            logger.warning("Segmentation mask has unexpected shape %s; returning zeros.", mask.shape)
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
        if mask.shape != image.shape[:2]:
            try:
                mask = cv2.resize(mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
            except Exception as exc:  # pragma: no cover - best effort resize
                logger.warning("Failed to resize mask from %s to %s (%s). Using zeros.", mask.shape, image.shape[:2], exc)
                mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)
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
    mask = np.asarray(mask)
    if mask.ndim == 3:
        mask = mask.mean(axis=-1)
    if mask.ndim != 2:
        logger.warning("Point mask weights received invalid mask shape %s; returning zeros.", mask.shape)
        return np.zeros(points_world.shape[0], dtype=np.float32)

    pix = project_points_to_pixels(points_world, cam_to_world, intrinsics)
    u = pix[:, 0].round().astype(int)
    v = pix[:, 1].round().astype(int)
    depth = pix[:, 2]
    h, w = mask.shape
    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depth > 0)
    weights = np.zeros(points_world.shape[0], dtype=np.float32)
    weights[valid] = mask[v[valid], u[valid]]
    return weights
