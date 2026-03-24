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
    """Load LaMa model if not already loaded."""
    global _lama_model, _lama_device

    if _lama_model is not None and _lama_device == config.device:
        return

    import torch

    from illust2psd.utils.download import get_model_path

    device = config.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    model_path = get_model_path("lama")
    logger.info(f"Loading LaMa model on {device}...")

    try:
        # Try loading as a full model checkpoint
        checkpoint = torch.load(str(model_path), map_location=device, weights_only=False)

        if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
            # It's a training checkpoint with state_dict
            logger.info("LaMa checkpoint loaded (state_dict format)")
            _lama_model = checkpoint
        elif isinstance(checkpoint, torch.nn.Module):
            _lama_model = checkpoint
        else:
            # Raw state dict or other format
            _lama_model = checkpoint

        _lama_device = config.device
        logger.info("LaMa model loaded successfully")
    except Exception as e:
        logger.warning(f"Failed to load LaMa: {e}, will use OpenCV fallback")
        _lama_model = None


def _lama_inpaint(rgb: np.ndarray, mask: np.ndarray, config: PipelineConfig) -> np.ndarray:
    """LaMa inpainting via PyTorch."""
    import torch

    global _lama_model

    if _lama_model is None:
        raise RuntimeError("LaMa model not loaded")

    device = config.device
    if device == "mps" and not torch.backends.mps.is_available():
        device = "cpu"

    h, w = rgb.shape[:2]

    # Prepare input tensors
    # LaMa expects: image (B, 3, H, W) in [0, 1], mask (B, 1, H, W) in {0, 1}
    # Pad to multiple of 8 for the network
    pad_h = (8 - h % 8) % 8
    pad_w = (8 - w % 8) % 8

    img_t = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    mask_t = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).unsqueeze(0)

    if pad_h > 0 or pad_w > 0:
        img_t = torch.nn.functional.pad(img_t, (0, pad_w, 0, pad_h), mode="reflect")
        mask_t = torch.nn.functional.pad(mask_t, (0, pad_w, 0, pad_h), mode="constant", value=0)

    img_t = img_t.to(device)
    mask_t = mask_t.to(device)

    try:
        if isinstance(_lama_model, torch.nn.Module):
            _lama_model.eval()
            with torch.no_grad():
                batch = {"image": img_t, "mask": mask_t}
                result = _lama_model(batch)
                if isinstance(result, dict):
                    out_t = result.get("inpainted", result.get("predicted_image", img_t))
                else:
                    out_t = result
        else:
            # For state_dict-based models, fall back to OpenCV
            raise RuntimeError("LaMa model format not directly runnable, using OpenCV")
    except Exception:
        raise

    # Convert back
    out = out_t[0].cpu().permute(1, 2, 0).numpy()
    out = (np.clip(out, 0, 1) * 255).astype(np.uint8)

    # Remove padding
    if pad_h > 0 or pad_w > 0:
        out = out[:h, :w]

    return out
