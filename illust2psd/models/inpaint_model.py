"""Inpainting model abstraction."""

from __future__ import annotations

from enum import Enum

import cv2
import numpy as np
from loguru import logger
from PIL import Image


class InpaintBackend(Enum):
    LAMA = "lama"
    OPENCV = "opencv"
    NONE = "none"


class Inpainter:
    """Lazy-loaded inpainting model."""

    def __init__(self, backend: str = "opencv") -> None:
        self.backend = InpaintBackend(backend)
        self._lama = None

    def _load_lama(self):
        if self._lama is not None:
            return

        from simple_lama_inpainting import SimpleLama

        self._lama = SimpleLama()
        logger.info("LaMa inpainting model loaded")

    def inpaint(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Inpaint masked regions.

        Args:
            rgb: RGB uint8 array (H, W, 3)
            mask: Boolean mask (H, W), True = region to inpaint

        Returns:
            Inpainted RGB uint8 array (H, W, 3)
        """
        if self.backend == InpaintBackend.NONE:
            return rgb.copy()

        if self.backend == InpaintBackend.LAMA:
            try:
                return self._inpaint_lama(rgb, mask)
            except Exception as e:
                logger.warning(f"LaMa failed: {e}, falling back to OpenCV")

        return self._inpaint_opencv(rgb, mask)

    def _inpaint_lama(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        self._load_lama()
        rgb_pil = Image.fromarray(rgb)
        mask_pil = Image.fromarray(mask.astype(np.uint8) * 255)
        result = self._lama(rgb_pil, mask_pil)
        return np.array(result)

    def _inpaint_opencv(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        mask_u8 = mask.astype(np.uint8) * 255
        return cv2.inpaint(rgb, mask_u8, inpaintRadius=3, flags=cv2.INPAINT_TELEA)
