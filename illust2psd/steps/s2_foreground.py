"""S2: Foreground Extraction — Remove background, produce binary foreground mask."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from PIL import Image
from scipy import ndimage

from illust2psd.config import PipelineConfig
from illust2psd.utils.mask_utils import close_mask, fill_holes


@dataclass
class ForegroundResult:
    """Output of the foreground extraction step."""

    mask: np.ndarray  # Boolean mask (H, W), True = foreground
    method: str  # Which method was used


def extract_foreground(
    image: Image.Image,
    has_transparent_bg: bool,
    config: PipelineConfig,
) -> ForegroundResult:
    """Extract foreground mask from image.

    Strategy (tiered):
    1. If already transparent: derive from alpha channel
    2. Primary: anime-segmentation ISNet (ONNX)
    3. Fallback: rembg with u2net
    4. Emergency: GrabCut

    Args:
        image: RGBA PIL Image
        has_transparent_bg: Whether background is already transparent
        config: Pipeline configuration

    Returns:
        ForegroundResult with binary mask
    """
    arr = np.array(image)

    # Strategy 1: Use existing alpha channel
    if has_transparent_bg:
        alpha_mask = arr[:, :, 3] > 10
        alpha_ratio = np.sum(alpha_mask) / max(1, alpha_mask.size)

        if alpha_ratio > 0.70:
            # Alpha covers >70% of canvas — likely a baked-in scene background.
            # Run ISNet on the RGB to try separating character from scene.
            logger.warning(
                f"Alpha covers {alpha_ratio:.0%} of canvas — possible baked-in scene. "
                "Running ISNet to refine foreground."
            )
            try:
                isnet_mask = _isnet_segment(arr, config)
                isnet_mask = _postprocess_mask(isnet_mask, config)
                # Use the intersection: ISNet foreground AND alpha foreground
                refined = isnet_mask & alpha_mask
                refined_ratio = np.sum(refined) / max(1, alpha_mask.size)
                if refined_ratio > 0.05:  # ISNet found something reasonable
                    logger.info(f"ISNet refined foreground: {alpha_ratio:.0%} → {refined_ratio:.0%}")
                    return ForegroundResult(mask=refined, method="alpha+isnet")
                else:
                    logger.info("ISNet refinement too aggressive, using alpha as-is")
            except Exception as e:
                logger.warning(f"ISNet refinement failed: {e}, using alpha as-is")

        logger.info("Using existing alpha channel for foreground mask")
        mask = _postprocess_mask(alpha_mask, config)
        return ForegroundResult(mask=mask, method="alpha")

    # Strategy 2: ISNet anime segmentation
    if config.foreground_model == "isnet":
        try:
            mask = _isnet_segment(arr, config)
            mask = _postprocess_mask(mask, config)
            logger.info("Foreground extracted with ISNet")
            return ForegroundResult(mask=mask, method="isnet")
        except Exception as e:
            logger.warning(f"ISNet failed: {e}, falling back to rembg")

    # Strategy 3: rembg
    if config.foreground_model in ("isnet", "rembg"):
        try:
            mask = _rembg_segment(image)
            mask = _postprocess_mask(mask, config)
            logger.info("Foreground extracted with rembg")
            return ForegroundResult(mask=mask, method="rembg")
        except Exception as e:
            logger.warning(f"rembg failed: {e}, falling back to GrabCut")

    # Strategy 4: GrabCut
    mask = _grabcut_segment(arr)
    mask = _postprocess_mask(mask, config)
    logger.info("Foreground extracted with GrabCut")
    return ForegroundResult(mask=mask, method="grabcut")


def _isnet_segment(arr: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Run ISNet anime segmentation via ONNX Runtime."""
    from illust2psd.models.model_manager import ModelManager

    session = ModelManager.get().get_isnet_session(config.device)

    # Prepare input: resize to 1024x1024, normalize
    h, w = arr.shape[:2]
    input_size = 1024

    # Use RGB only
    rgb = arr[:, :, :3].astype(np.float32) / 255.0
    # Resize
    resized = cv2.resize(rgb, (input_size, input_size), interpolation=cv2.INTER_LINEAR)
    # Normalize to [-1, 1]
    resized = (resized - 0.5) / 0.5
    # NCHW format
    tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, :].astype(np.float32)

    input_name = session.get_inputs()[0].name
    output = session.run(None, {input_name: tensor})[0]

    # Output is (1, 1, H, W) or (1, H, W) — sigmoid probabilities
    pred = output.squeeze()
    if pred.ndim == 3:
        pred = pred[0]

    # Resize back to original size
    pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
    mask = pred > 0.5
    return mask


def _rembg_segment(image: Image.Image) -> np.ndarray:
    """Use rembg library for background removal."""
    from rembg import remove

    result = remove(image)
    result_arr = np.array(result)
    mask = result_arr[:, :, 3] > 10
    return mask


def _grabcut_segment(arr: np.ndarray) -> np.ndarray:
    """OpenCV GrabCut fallback — no model needed."""
    rgb = arr[:, :, :3]
    h, w = rgb.shape[:2]

    mask = np.zeros((h, w), np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)

    # Initial rectangle: assume character is centered with some margin
    margin_x = int(w * 0.05)
    margin_y = int(h * 0.02)
    rect = (margin_x, margin_y, w - 2 * margin_x, h - 2 * margin_y)

    cv2.grabCut(rgb, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)

    # Convert GrabCut mask to binary
    fg_mask = np.isin(mask, [cv2.GC_FGD, cv2.GC_PR_FGD])
    return fg_mask


def _postprocess_mask(mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """Clean up foreground mask."""
    from illust2psd.utils.mask_utils import remove_small_components

    # Morphological close (fills small gaps)
    mask = close_mask(mask, config.mask_close_kernel)
    # Additional close with larger kernel to seal ISNet edge gaps on dark backgrounds
    mask = close_mask(mask, max(config.mask_close_kernel, 7))
    # Fill interior holes
    mask = fill_holes(mask)
    # Remove small disconnected fragments (ISNet noise on colored backgrounds)
    h, w = mask.shape
    min_fg_px = max(200, int(h * w * 0.001))  # At least 0.1% of image
    mask = remove_small_components(mask, min_pixels=min_fg_px)
    return mask.astype(bool)
