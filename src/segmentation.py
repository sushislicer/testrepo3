"""CLIPSeg affordance masking and 2D-to-3D projection utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np
import torch

from .config import SegmentationConfig

logger = logging.getLogger(__name__)


class CLIPSegSegmenter:
    """Wrapper for CLIPSeg segmentation with lazy model loading and auto-cropping."""

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

    def _get_auto_crop_box(self, image: np.ndarray, padding: int = 20) -> Optional[Tuple[int, int, int, int]]:
        """
        Detect object bounding box to focus CLIPSeg.
        Returns (x_min, y_min, x_max, y_max) or None if empty.
        """
        # Convert to grayscale if needed for thresholding
        if image.ndim == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image

        # Assuming object is non-black against black background (or invert if white bg)
        # Using a low threshold to catch dark objects.
        coords = cv2.findNonZero(gray)
        if coords is None:
            return None

        x, y, w, h = cv2.boundingRect(coords)
        
        # Add padding
        img_h, img_w = image.shape[:2]
        x_min = max(0, x - padding)
        y_min = max(0, y - padding)
        x_max = min(img_w, x + w + padding)
        y_max = min(img_h, y + h + padding)

        # Make square for better CLIPSeg/Point-E consistency
        crop_w = x_max - x_min
        crop_h = y_max - y_min
        diff = crop_w - crop_h
        if diff > 0: # Wider than tall
            pad_v = diff // 2
            y_min = max(0, y_min - pad_v)
            y_max = min(img_h, y_max + pad_v)
        elif diff < 0: # Taller than wide
            pad_h = abs(diff) // 2
            x_min = max(0, x_min - pad_h)
            x_max = min(img_w, x_max + pad_h)

        return (x_min, y_min, x_max, y_max)

    def segment_affordance(self, image: np.ndarray, prompt: Optional[str] = None) -> np.ndarray:
        """
        Return a float mask in [0,1] with same HxW as input image.
        Performs auto-cropping to improve resolution on small objects.
        """
        if not self.available or self.model is None or self.processor is None:
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.float32)

        prompt = prompt or self.cfg.prompt
        full_h, full_w = image.shape[:2]
        
        # 1. Determine Crop
        crop_box = self._get_auto_crop_box(image)
        if crop_box:
            x1, y1, x2, y2 = crop_box
            input_image = image[y1:y2, x1:x2]
        else:
            x1, y1, x2, y2 = 0, 0, full_w, full_h
            input_image = image

        # 2. Run Inference on (Cropped) Image
        inputs = self.processor(
            text=[prompt],
            images=[input_image],
            return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = torch.sigmoid(logits)

        # Handle dimensions
        if probs.dim() == 4: probs = probs[0, 0]
        elif probs.dim() == 3: probs = probs[0]
        
        mask_crop = probs.detach().cpu().numpy().astype(np.float32)
        
        # 3. Resize mask to match the Crop Size
        crop_h, crop_w = (y2 - y1), (x2 - x1)
        if mask_crop.shape != (crop_h, crop_w):
             mask_crop = cv2.resize(mask_crop, (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

        # 4. Paste back into Full Size Frame ("Un-crop")
        # This ensures the mask aligns with global camera intrinsics
        full_mask = np.zeros((full_h, full_w), dtype=np.float32)
        full_mask[y1:y2, x1:x2] = mask_crop

        return full_mask


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
    
    # Validation
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


def compute_combined_score(
    variance_grid: np.ndarray, 
    semantic_weights: np.ndarray
) -> np.ndarray:
    """
    Compute the final voxel score by combining variance and semantic weights.
    Formula: Score(v) = Variance(v) * SemanticWeight(v)
    """
    raw_scores = variance_grid * semantic_weights
    
    # Normalize scores to [0, 1] for easier visualization/thresholding
    min_val = np.min(raw_scores)
    max_val = np.max(raw_scores)
    
    if max_val - min_val < 1e-6:
        return np.zeros_like(raw_scores)
        
    normalized_scores = (raw_scores - min_val) / (max_val - min_val)
    return normalized_scores