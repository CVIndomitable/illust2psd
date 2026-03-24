"""S1: Preprocess — Load image, validate, normalize to RGBA, resize."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from illust2psd.config import PipelineConfig
from illust2psd.utils.image_utils import auto_orient, ensure_rgba, has_transparent_background, resize_to_max


@dataclass
class PreprocessResult:
    """Output of the preprocess step."""

    image: Image.Image  # RGBA PIL Image at working resolution
    original_size: tuple[int, int]  # (width, height) of original
    working_size: tuple[int, int]  # (width, height) after resize
    has_transparent_bg: bool  # True if background is already transparent


def preprocess(image_path: str | Path, config: PipelineConfig) -> PreprocessResult:
    """Load and preprocess an image for the pipeline.

    Args:
        image_path: Path to input image (PNG, JPG, WEBP)
        config: Pipeline configuration

    Returns:
        PreprocessResult with normalized RGBA image and metadata

    Raises:
        ValueError: If image dimensions are out of acceptable range
        FileNotFoundError: If image file doesn't exist
    """
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")

    logger.info(f"Loading image: {path}")
    img = Image.open(path)

    # Auto-orient from EXIF
    img = auto_orient(img)
    original_size = img.size  # (w, h)

    # Validate dimensions
    w, h = original_size
    min_s = config.min_image_size
    max_s = config.max_image_size
    if w < min_s or h < min_s:
        raise ValueError(f"Image too small: {w}x{h}, minimum side is {min_s}px")
    if w > max_s or h > max_s:
        raise ValueError(f"Image too large: {w}x{h}, maximum side is {max_s}px")

    # Convert to RGBA
    img = ensure_rgba(img)

    # Check if background is already transparent
    transparent_bg = has_transparent_background(img)
    if transparent_bg:
        logger.info("Detected transparent background — will skip foreground extraction")

    # Resize for processing
    img = resize_to_max(img, config.max_working_size)
    working_size = img.size

    logger.info(f"Preprocessed: {original_size[0]}x{original_size[1]} → {working_size[0]}x{working_size[1]}")

    return PreprocessResult(
        image=img,
        original_size=original_size,
        working_size=working_size,
        has_transparent_bg=transparent_bg,
    )
