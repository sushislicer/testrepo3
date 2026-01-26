"""CLIPSeg affordance masking utilities.

This module supports two related operations:

1) 2D affordance segmentation via CLIPSeg.
2) 3D-to-2D projection helpers that let us "paint" a Point-E point cloud by
   rendering it into 2D views, running CLIPSeg on those renders, and projecting
   the resulting masks back onto 3D points.
"""

from __future__ import annotations

import logging
import math
from typing import Iterable, Optional, Tuple

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

            self.processor = CLIPSegProcessor.from_pretrained(self.cfg.model_name, use_fast=True)
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


def _rotation_z(deg: float) -> np.ndarray:
    """3x3 rotation matrix around +Z (degrees)."""
    rad = math.radians(deg)
    c, s = math.cos(rad), math.sin(rad)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def rotate_cam_to_world_yaw(cam_to_world: np.ndarray, yaw_deg: float) -> np.ndarray:
    """Rotate a camera pose around the world Z axis (about the origin)."""
    rot = np.eye(4, dtype=np.float32)
    rot[:3, :3] = _rotation_z(yaw_deg)
    return rot @ cam_to_world


def project_points_to_pixels(
    points_world: np.ndarray,
    cam_to_world: np.ndarray,
    intrinsics,
) -> np.ndarray:
    """Project 3D points (N,3) in world frame to pixel coords (N,2) using intrinsics and pose."""
    world_to_cam = np.linalg.inv(cam_to_world)
    homog = np.concatenate([points_world, np.ones((points_world.shape[0], 1), dtype=np.float32)], axis=1)
    cam = (world_to_cam @ homog.T).T[:, :3]

    # Camera convention: poses come from pyrender/OpenGL where the camera looks down -Z.
    # That means points "in front" have cam_z < 0. Use depth = -cam_z for positive depth.
    depth = (-cam[:, 2]).astype(np.float32)
    depth_safe = np.clip(depth, 1e-6, None)

    u = intrinsics.fx * cam[:, 0] / depth_safe + intrinsics.cx
    # cam_y is up, but image v increases downwards -> flip sign.
    v = -intrinsics.fy * cam[:, 1] / depth_safe + intrinsics.cy

    return np.stack([u, v, depth], axis=1)  # (u, v, depth>0 in front)


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


def render_point_cloud_view(
    points_world: np.ndarray,
    cam_to_world: np.ndarray,
    intrinsics,
    *,
    background: int = 0,
    point_radius_px: int = 2,
    blur_sigma: float = 0.8,
) -> np.ndarray:
    """
    Lightweight, dependency-free point cloud renderer.

    Produces a 3-channel uint8 image suitable as input to CLIPSeg by:
    - projecting points to pixels
    - z-buffering with a per-pixel min depth
    - mapping depth to intensity (closer -> brighter)
    - optionally dilating + blurring to reduce sparsity
    """
    h = int(getattr(intrinsics, "height", 512))
    w = int(getattr(intrinsics, "width", 512))

    if points_world.size == 0:
        return np.full((h, w, 3), background, dtype=np.uint8)

    pix = project_points_to_pixels(points_world, cam_to_world, intrinsics)
    u = pix[:, 0].round().astype(np.int32)
    v = pix[:, 1].round().astype(np.int32)
    depth = pix[:, 2]

    valid = (u >= 0) & (u < w) & (v >= 0) & (v < h) & (depth > 0)
    if not np.any(valid):
        return np.full((h, w, 3), background, dtype=np.uint8)

    u = u[valid]
    v = v[valid]
    depth = depth[valid].astype(np.float32)

    depth_map = np.full((h, w), np.inf, dtype=np.float32)
    np.minimum.at(depth_map, (v, u), depth)

    occ = np.isfinite(depth_map)
    img = np.zeros((h, w), dtype=np.float32)
    if np.any(occ):
        d = depth_map[occ]
        d_min = float(d.min())
        d_max = float(d.max())
        denom = (d_max - d_min) if (d_max - d_min) > 1e-6 else 1e-6
        # Closer -> brighter
        img[occ] = (d_max - depth_map[occ]) / denom

    img_u8 = (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)

    if point_radius_px > 0:
        k = 2 * int(point_radius_px) + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        img_u8 = cv2.dilate(img_u8, kernel, iterations=1)

    if blur_sigma and blur_sigma > 0:
        img_u8 = cv2.GaussianBlur(img_u8, ksize=(0, 0), sigmaX=float(blur_sigma))

    if background != 0:
        bg = np.uint8(int(background))
        img_u8 = np.where(img_u8 > 0, img_u8, bg).astype(np.uint8)

    return np.repeat(img_u8[..., None], 3, axis=2)


def compute_point_mask_weights_multiview(
    points_world: np.ndarray,
    base_cam_to_world: np.ndarray,
    intrinsics,
    segmenter,
    *,
    prompt: Optional[str] = None,
    num_views: int = 3,
    yaw_step_deg: Optional[float] = None,
    aggregation: str = "max",
    point_radius_px: int = 2,
    blur_sigma: float = 0.8,
) -> np.ndarray:
    """
    Compute per-point semantic weights by rendering `num_views` around the cloud,
    running CLIPSeg per-render, and projecting the 2D masks back to 3D points.

    `base_cam_to_world` is used for the first view (intended to match the input view).
    Additional views rotate the camera around world Z by `yaw_step_deg`.
    """
    if num_views < 1:
        raise ValueError("num_views must be >= 1")

    if yaw_step_deg is None:
        yaw_step_deg = 360.0 / float(num_views)

    weights_per_view = []
    for i in range(num_views):
        yaw = float(i) * float(yaw_step_deg)
        cam_to_world = rotate_cam_to_world_yaw(base_cam_to_world, yaw)

        render = render_point_cloud_view(
            points_world,
            cam_to_world,
            intrinsics,
            point_radius_px=point_radius_px,
            blur_sigma=blur_sigma,
        )
        mask = segmenter.segment_affordance(render, prompt=prompt)
        weights = compute_point_mask_weights(points_world, cam_to_world, intrinsics, mask)
        weights_per_view.append(weights)

    if not weights_per_view:
        return np.zeros(points_world.shape[0], dtype=np.float32)

    stack = np.stack(weights_per_view, axis=0)

    aggregation = aggregation.lower().strip()
    if aggregation == "max":
        return np.max(stack, axis=0)
    if aggregation == "mean":
        return np.mean(stack, axis=0)
    if aggregation == "sum":
        return np.sum(stack, axis=0)
    raise ValueError(f"Unknown aggregation '{aggregation}' (expected: max|mean|sum)")


