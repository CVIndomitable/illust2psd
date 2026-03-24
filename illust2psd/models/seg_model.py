"""Segmentation model abstraction — lazy loading and inference wrappers."""

from __future__ import annotations

from enum import Enum

import numpy as np
from loguru import logger


class SegBackend(Enum):
    ISNET = "isnet"
    REMBG = "rembg"
    GRABCUT = "grabcut"


class ForegroundSegmenter:
    """Lazy-loaded foreground segmentation model."""

    def __init__(self, backend: str = "isnet", device: str = "cpu") -> None:
        self.backend = SegBackend(backend)
        self.device = device
        self._session = None

    def _load_isnet(self):
        """Load ISNet ONNX session."""
        if self._session is not None:
            return

        import onnxruntime as ort

        from illust2psd.utils.download import get_model_path

        model_path = get_model_path("isnet_anime")
        providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if self.device == "cpu":
            providers = ["CPUExecutionProvider"]

        self._session = ort.InferenceSession(str(model_path), providers=providers)
        logger.info(f"ISNet loaded on {self._session.get_providers()}")

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        """Run foreground prediction.

        Args:
            rgb: RGB uint8 array (H, W, 3)

        Returns:
            Probability map (H, W) float32 in [0, 1]
        """
        if self.backend == SegBackend.ISNET:
            return self._predict_isnet(rgb)
        raise NotImplementedError(f"Backend {self.backend} not supported in model wrapper")

    def _predict_isnet(self, rgb: np.ndarray) -> np.ndarray:
        """ISNet inference."""
        import cv2

        self._load_isnet()

        h, w = rgb.shape[:2]
        input_size = 1024

        resized = cv2.resize(rgb.astype(np.float32) / 255.0, (input_size, input_size))
        resized = (resized - 0.5) / 0.5
        tensor = np.transpose(resized, (2, 0, 1))[np.newaxis, :].astype(np.float32)

        input_name = self._session.get_inputs()[0].name
        output = self._session.run(None, {input_name: tensor})[0]

        pred = output.squeeze()
        if pred.ndim == 3:
            pred = pred[0]

        pred = cv2.resize(pred, (w, h))
        return pred
