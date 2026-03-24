"""Test fixtures for illust2psd."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_rgba_image() -> Image.Image:
    """Create a simple 256x256 RGBA test image with a character-like shape."""
    img = np.zeros((256, 256, 4), dtype=np.uint8)
    # Simple character silhouette: head (circle) + body (rectangle)
    # Head
    for y in range(40, 90):
        for x in range(103, 153):
            if (x - 128) ** 2 + (y - 65) ** 2 < 25 ** 2:
                img[y, x] = [200, 180, 160, 255]  # Skin color
    # Body
    img[90:180, 108:148] = [100, 100, 200, 255]  # Blue clothing
    # Arms
    img[95:140, 88:108] = [200, 180, 160, 255]  # Left arm
    img[95:140, 148:168] = [200, 180, 160, 255]  # Right arm
    # Legs
    img[180:230, 110:128] = [80, 80, 160, 255]  # Left leg
    img[180:230, 128:146] = [80, 80, 160, 255]  # Right leg
    # Hair (dark)
    for y in range(30, 70):
        for x in range(98, 158):
            if (x - 128) ** 2 + (y - 55) ** 2 < 30 ** 2:
                img[y, x] = [40, 30, 50, 255]

    return Image.fromarray(img, "RGBA")


@pytest.fixture
def sample_image_path(sample_rgba_image: Image.Image, tmp_path: Path) -> Path:
    """Save the sample image to a temp file and return the path."""
    path = tmp_path / "test_character.png"
    sample_rgba_image.save(path)
    return path


@pytest.fixture
def transparent_bg_image() -> Image.Image:
    """Create a test image with transparent background."""
    img = np.zeros((256, 256, 4), dtype=np.uint8)
    # Character in the center with transparent background
    img[60:200, 80:176] = [180, 160, 140, 255]
    return Image.fromarray(img, "RGBA")
