"""Global model manager — lazy loading with caching to avoid repeated initialization."""

from __future__ import annotations

import os

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
        self._segformer_processor = None
        self._segformer_model = None
        self._gdino_processor = None
        self._gdino_model = None

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

    def get_segformer(self) -> tuple:
        """Get or create SegFormer human parsing model (processor, model).

        Uses mattmdjaga/segformer_b2_clothes (ATR dataset, 18 classes).
        """
        if self._segformer_processor is not None:
            return self._segformer_processor, self._segformer_model

        from transformers import SegformerForSemanticSegmentation, SegformerImageProcessor

        model_id = "mattmdjaga/segformer_b2_clothes"
        logger.info(f"Loading SegFormer ({model_id})...")

        # Set proxy env for HuggingFace download if local proxy is available
        import socket
        try:
            sock = socket.create_connection(("127.0.0.1", 7897), timeout=1)
            sock.close()
            os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
            os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
        except (ConnectionRefusedError, OSError, socket.timeout):
            pass

        self._segformer_processor = SegformerImageProcessor.from_pretrained(model_id)
        self._segformer_model = SegformerForSemanticSegmentation.from_pretrained(model_id)
        self._segformer_model.eval()

        logger.info("SegFormer ready (ATR 18-class)")
        return self._segformer_processor, self._segformer_model

    def get_grounding_dino(self) -> tuple:
        """Get or create Grounding DINO model for open-vocab object detection.

        Uses IDEA-Research/grounding-dino-tiny (~250MB).
        Returns (processor, model).
        """
        if self._gdino_processor is not None:
            return self._gdino_processor, self._gdino_model

        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        model_id = "IDEA-Research/grounding-dino-tiny"
        logger.info(f"Loading Grounding DINO ({model_id})...")

        import socket
        try:
            sock = socket.create_connection(("127.0.0.1", 7897), timeout=1)
            sock.close()
            os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7897")
            os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7897")
        except (ConnectionRefusedError, OSError, socket.timeout):
            pass

        self._gdino_processor = AutoProcessor.from_pretrained(model_id)
        self._gdino_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id)
        self._gdino_model.eval()

        logger.info("Grounding DINO ready")
        return self._gdino_processor, self._gdino_model

    def clear(self) -> None:
        """Release all loaded models."""
        self._sam2_predictor = None
        self._sam2_device = None
        self._isnet_session = None
        self._isnet_device = None
        self._segformer_processor = None
        self._segformer_model = None
        self._gdino_processor = None
        self._gdino_model = None
        logger.info("All models released")
