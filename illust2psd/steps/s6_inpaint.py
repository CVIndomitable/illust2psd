"""S6: Inpainting — Complete occluded regions behind each part.

Phase 2: LaMa inpainting via direct PyTorch loading, with OpenCV fallback.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from illust2psd.config import PipelineConfig, get_z_order_map
from illust2psd.utils.mask_utils import dilate_mask, feather_edges


# Global LaMa model cache to avoid reloading
_lama_model = None
_lama_device = None


def inpaint_parts(
    image: Image.Image,
    masks: dict[str, np.ndarray],
    full_masks: dict[str, np.ndarray],
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Inpaint occluded regions for each part.

    For each part, identifies pixels that are occluded by higher-z parts
    and fills them in.

    Args:
        image: RGBA PIL Image
        masks: Visible masks (after overlap removal)
        full_masks: Full masks (before overlap removal)
        config: Pipeline configuration

    Returns:
        Dict of part_id -> RGBA uint8 array (full image size, transparent outside part)
    """
    if config.inpaint_backend == "none":
        logger.info("Inpainting disabled, using visible regions only")
        return _no_inpaint(image, masks, config)

    arr = np.array(image)
    h, w = arr.shape[:2]
    z_map = get_z_order_map()
    part_images = {}

    # Pre-load LaMa if needed
    if config.inpaint_backend == "lama":
        _ensure_lama_loaded(config)

    inpaint_count = 0
    total_inpainted_px = 0

    for part_id, visible_mask in masks.items():
        full_mask = full_masks.get(part_id, visible_mask)

        # Occluded region: belongs to this part but covered by higher-z parts
        occluded = full_mask & ~visible_mask

        if not np.any(occluded):
            part_rgba = _extract_with_mask(arr, visible_mask, config)
            part_images[part_id] = part_rgba
            continue

        # Expand inpaint mask slightly for better blending
        occluded_expanded = dilate_mask(occluded, config.inpaint_expand_px)

        # Inpaint
        if config.inpaint_backend == "lama":
            try:
                inpainted = _lama_inpaint(arr[:, :, :3], occluded_expanded, config)
            except Exception as e:
                logger.warning(f"LaMa failed for {part_id}: {e}, using OpenCV")
                inpainted = _opencv_inpaint(arr[:, :, :3], occluded_expanded)
        else:
            inpainted = _opencv_inpaint(arr[:, :, :3], occluded_expanded)

        # Composite: use original pixels where visible, inpainted where occluded
        part_rgb = arr[:, :, :3].copy()
        part_rgb[occluded] = inpainted[occluded]

        # Build RGBA with feathered alpha from full_mask
        alpha_float = feather_edges(full_mask, config.mask_feather_radius)
        alpha = (alpha_float * 255).astype(np.uint8)

        # Also respect original alpha
        if arr.shape[2] == 4:
            alpha = np.minimum(alpha, arr[:, :, 3])

        part_rgba = np.dstack([part_rgb, alpha])
        part_images[part_id] = part_rgba

        inpaint_count += 1
        total_inpainted_px += int(np.sum(occluded))

    logger.info(f"Inpainting complete: {inpaint_count} parts, {total_inpainted_px} pixels filled")
    return part_images


def _no_inpaint(
    image: Image.Image,
    masks: dict[str, np.ndarray],
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Extract parts without inpainting."""
    arr = np.array(image)
    result = {}
    for part_id, mask in masks.items():
        result[part_id] = _extract_with_mask(arr, mask, config)
    return result


def _extract_with_mask(
    arr: np.ndarray,
    mask: np.ndarray,
    config: PipelineConfig,
) -> np.ndarray:
    """Extract RGBA image using a boolean mask with feathered edges."""
    h, w = arr.shape[:2]
    result = np.zeros((h, w, 4), dtype=np.uint8)
    result[:, :, :3] = arr[:, :, :3]

    # Feathered alpha
    alpha_float = feather_edges(mask, config.mask_feather_radius)
    result[:, :, 3] = (alpha_float * 255).astype(np.uint8)

    # Respect original alpha
    if arr.shape[2] == 4:
        result[:, :, 3] = np.minimum(result[:, :, 3], arr[:, :, 3])

    return result


def _opencv_inpaint(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """OpenCV Telea inpainting — fast but low quality."""
    mask_u8 = mask.astype(np.uint8) * 255
    inpainted = cv2.inpaint(rgb, mask_u8, inpaintRadius=5, flags=cv2.INPAINT_TELEA)
    return inpainted


def _ensure_lama_loaded(config: PipelineConfig) -> None:
    """Load LaMa model via simple-lama-inpainting."""
    global _lama_model, _lama_device

    if _lama_model is not None and _lama_device == config.device:
        return

    try:
        from simple_lama_inpainting import SimpleLama
        logger.info("Loading LaMa model (simple-lama-inpainting)...")
        _lama_model = SimpleLama()
        _lama_device = config.device
        logger.info("LaMa model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load LaMa: {e}, will use OpenCV fallback")
        _lama_model = None


def _lama_inpaint(rgb: np.ndarray, mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """LaMa inpainting via simple-lama-inpainting."""
    from simple_lama_inpainting import SimpleLama

    global _lama_model

    if _lama_model is None or not isinstance(_lama_model, SimpleLama):
        raise RuntimeError("LaMa model not loaded")

    img_pil = Image.fromarray(rgb)
    mask_pil = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")

    result_pil = _lama_model(img_pil, mask_pil)
    return np.array(result_pil.convert("RGB"))
