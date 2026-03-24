"""Pose estimation model abstraction."""

from __future__ import annotations

from enum import Enum

import numpy as np
from loguru import logger


class PoseBackend(Enum):
    MEDIAPIPE = "mediapipe"
    HEURISTIC = "heuristic"


class PoseEstimator:
    """Lazy-loaded pose estimation model."""

    def __init__(self, backend: str = "mediapipe") -> None:
        self.backend = PoseBackend(backend)
        self._pose = None

    def _load_mediapipe(self):
        if self._pose is not None:
            return

        import mediapipe as mp

        self._pose = mp.solutions.pose.Pose(
            static_image_mode=True,
            model_complexity=2,
            min_detection_confidence=0.3,
        )
        logger.info("MediaPipe Pose loaded")

    def predict(self, rgb: np.ndarray) -> dict | None:
        """Run pose estimation.

        Args:
            rgb: RGB uint8 array (H, W, 3)

        Returns:
            MediaPipe pose_landmarks or None
        """
        if self.backend == PoseBackend.MEDIAPIPE:
            self._load_mediapipe()
            results = self._pose.process(rgb)
            return results.pose_landmarks if results.pose_landmarks else None
        return None

    def close(self) -> None:
        if self._pose is not None:
            self._pose.close()
            self._pose = None
