"""Global model manager — lazy loading with caching to avoid repeated initialization."""

from __future__ import annotations

from loguru import logger


class ModelManager:
    """Singleton manager for all ML models used in the pipeline.

    Each model is loaded on first access and cached for reuse.
    Call `clear()` to release all models from memory.
    """

    _instance: ModelManager | None = None

    def __init__(self) -> None:
        self._sam2_predictor = None
        self._sam2_device = None
        self._isnet_session = None
        self._isnet_device = None

    @classmethod
    def get(cls) -> ModelManager:
        if cls._instance is None:
            cls._instance = ModelManager()
        return cls._instance

    def get_sam2_predictor(self, device: str = "mps"):
        """Get or create SAM2 image predictor."""
        if self._sam2_predictor is not None and self._sam2_device == device:
            return self._sam2_predictor

        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        from illust2psd.utils.download import get_model_path

        if device == "mps" and not torch.backends.mps.is_available():
            device = "cpu"

        model_path = get_model_path("sam2_hiera_large")
        logger.info(f"Loading SAM2 on {device}...")

        sam2 = build_sam2(
            "configs/sam2.1/sam2.1_hiera_l.yaml",
            str(model_path),
            device=device,
        )
        self._sam2_predictor = SAM2ImagePredictor(sam2)
        self._sam2_device = device

        logger.info("SAM2 ready")
        return self._sam2_predictor

    def get_isnet_session(self, device: str = "mps"):
        """Get or create ISNet ONNX inference session."""
        if self._isnet_session is not None and self._isnet_device == device:
            return self._isnet_session

        import onnxruntime as ort

        from illust2psd.utils.download import get_model_path

        model_path = get_model_path("isnet_anime")

        providers = ["CPUExecutionProvider"]
        if device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        elif device == "mps":
            providers = ["CoreMLExecutionProvider", "CPUExecutionProvider"]

        logger.info(f"Loading ISNet ({providers[0]})...")
        self._isnet_session = ort.InferenceSession(str(model_path), providers=providers)
        self._isnet_device = device

        logger.info("ISNet ready")
        return self._isnet_session

    def clear(self) -> None:
        """Release all loaded models."""
        self._sam2_predictor = None
        self._sam2_device = None
        self._isnet_session = None
        self._isnet_device = None
        logger.info("All models released")
