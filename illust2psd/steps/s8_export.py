"""S8: PSD Export — Write Cubism-compliant layered PSD file."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from illust2psd.config import PipelineConfig
from illust2psd.steps.s7_compose import LayerSpec
from illust2psd.utils.psd_utils import create_psd


def export_psd(
    layers: list[LayerSpec],
    canvas_size: tuple[int, int],
    output_path: str | Path,
    config: PipelineConfig,
    original_image: Image.Image | None = None,
) -> Path:
    """Export layers as a Cubism-compliant PSD file.

    PSD Requirements for Cubism Editor:
    - Format: PSD (not PSB)
    - Color mode: RGB, 8-bit
    - Each part = one layer, Normal blending, 100% opacity
    - No layer masks, adjustment layers, or smart objects
    - No duplicate layer names
    - Canvas size = original illustration size

    Args:
        layers: List of LayerSpec sorted by z_order (back to front)
        canvas_size: (width, height) of the PSD canvas
        output_path: Where to write the PSD file
        config: Pipeline configuration
        original_image: Optional original image for reference layer

    Returns:
        Path to the written PSD file
    """
    output_path = Path(output_path)
    canvas_w, canvas_h = canvas_size

    logger.info(f"Exporting PSD: {canvas_w}x{canvas_h}, {len(layers)} layers")

    # Build layer specs for psd_utils
    layer_dicts = []
    for layer in layers:
        layer_dicts.append({
            "name": layer.name,
            "image": layer.image,
            "offset_x": layer.offset_x,
            "offset_y": layer.offset_y,
        })

    # Reference layer (flattened original, hidden)
    reference = None
    if config.include_reference_layer and original_image is not None:
        ref_arr = np.array(original_image)
        if ref_arr.shape[2] == 3:
            # Add alpha channel
            alpha = np.full((*ref_arr.shape[:2], 1), 255, dtype=np.uint8)
            ref_arr = np.concatenate([ref_arr, alpha], axis=2)
        reference = ref_arr

    create_psd(
        canvas_width=canvas_w,
        canvas_height=canvas_h,
        layer_specs=layer_dicts,
        output_path=output_path,
        reference_image=reference,
    )

    # Log file size
    file_size = output_path.stat().st_size
    logger.info(f"PSD written: {output_path} ({file_size / 1024 / 1024:.1f} MB)")

    return output_path
