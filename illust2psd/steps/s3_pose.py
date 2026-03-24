"""S3: Pose Estimation — Detect body keypoints for segmentation guidance."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from illust2psd.config import PipelineConfig, MODEL_CACHE_DIR


@dataclass
class Keypoint:
    """A single detected keypoint."""

    x: float  # Pixel coordinate
    y: float  # Pixel coordinate
    confidence: float


@dataclass
class PoseResult:
    """Output of pose estimation."""

    keypoints: dict[str, Keypoint]
    method: str

    def get(self, name: str) -> Keypoint | None:
        kp = self.keypoints.get(name)
        if kp and kp.confidence >= 0.3:
            return kp
        return None

    def midpoint(self, name_a: str, name_b: str) -> tuple[float, float] | None:
        a = self.get(name_a)
        b = self.get(name_b)
        if a and b:
            return ((a.x + b.x) / 2, (a.y + b.y) / 2)
        return None


# MediaPipe PoseLandmarker landmark indices to our keypoint names
_MP_LANDMARK_MAP = {
    0: "nose",
    2: "left_eye",      # left eye inner → approximate as left eye
    5: "right_eye",     # right eye inner → approximate as right eye
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}

_POSE_MODEL_URL = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/latest/pose_landmarker_heavy.task"
_POSE_MODEL_FILE = "pose_landmarker_heavy.task"


def estimate_pose(
    image: Image.Image,
    fg_mask: np.ndarray,
    config: PipelineConfig,
) -> PoseResult:
    """Estimate body keypoints.

    Strategy:
    1. Primary: MediaPipe PoseLandmarker (Tasks API)
    2. Fallback: heuristic proportions from foreground mask

    Args:
        image: RGBA PIL Image
        fg_mask: Boolean foreground mask (H, W)
        config: Pipeline configuration

    Returns:
        PoseResult with keypoints dict
    """
    if config.pose_backend == "mediapipe":
        try:
            result = _mediapipe_pose(image, config)
            if result and len(result.keypoints) > 5:
                logger.info(f"Pose estimated with MediaPipe ({len(result.keypoints)} keypoints)")
                return result
            logger.warning("MediaPipe returned too few keypoints, using heuristic")
        except Exception as e:
            logger.warning(f"MediaPipe failed: {e}, using heuristic fallback")

    result = _heuristic_pose(fg_mask)
    logger.info(f"Pose estimated with heuristic ({len(result.keypoints)} keypoints)")
    return result


def _ensure_pose_model() -> Path:
    """Download MediaPipe pose model if not cached."""
    model_path = MODEL_CACHE_DIR / _POSE_MODEL_FILE
    if model_path.exists():
        return model_path

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading MediaPipe pose model...")

    from illust2psd.utils.download import _download_url
    _download_url(_POSE_MODEL_URL, model_path, _POSE_MODEL_FILE)

    return model_path


def _mediapipe_pose(image: Image.Image, config: PipelineConfig) -> PoseResult | None:
    """Use MediaPipe PoseLandmarker (Tasks API) to detect keypoints."""
    import mediapipe as mp

    model_path = _ensure_pose_model()

    PoseLandmarker = mp.tasks.vision.PoseLandmarker
    PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
    BaseOptions = mp.tasks.BaseOptions

    options = PoseLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(model_path)),
        num_poses=1,
    )
    detector = PoseLandmarker.create_from_options(options)

    arr = np.array(image)
    rgb = arr[:, :, :3].copy()
    h, w = rgb.shape[:2]

    mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = detector.detect(mp_img)
    detector.close()

    if not result.pose_landmarks or len(result.pose_landmarks) == 0:
        return None

    landmarks = result.pose_landmarks[0]
    keypoints = {}
    for idx, name in _MP_LANDMARK_MAP.items():
        if idx < len(landmarks):
            lm = landmarks[idx]
            keypoints[name] = Keypoint(
                x=lm.x * w,
                y=lm.y * h,
                confidence=lm.visibility if hasattr(lm, 'visibility') and lm.visibility else lm.presence if hasattr(lm, 'presence') else 0.5,
            )

    return PoseResult(keypoints=keypoints, method="mediapipe")


def _heuristic_pose(fg_mask: np.ndarray) -> PoseResult:
    """Estimate keypoints from foreground mask using anime character proportions.

    Assumes a front-facing standing character.
    """
    h, w = fg_mask.shape
    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)

    if not rows.any():
        return PoseResult(keypoints={}, method="heuristic")

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]

    fg_h = y_max - y_min
    fg_w = x_max - x_min
    cx = (x_min + x_max) / 2

    head_top = y_min
    head_h = fg_h * 0.15
    head_bottom = head_top + head_h

    shoulder_y = head_bottom + fg_h * 0.03
    hip_y = y_min + fg_h * 0.52
    knee_y = y_min + fg_h * 0.75
    ankle_y = y_min + fg_h * 0.95
    elbow_y = y_min + fg_h * 0.38
    wrist_y = y_min + fg_h * 0.50

    shoulder_half = fg_w * 0.22
    hip_half = fg_w * 0.15
    arm_offset = fg_w * 0.30

    face_cy = head_top + head_h * 0.55
    eye_y = head_top + head_h * 0.45
    nose_y = head_top + head_h * 0.65
    eye_half = fg_w * 0.06
    ear_half = fg_w * 0.10

    kp = {
        "nose": Keypoint(cx, nose_y, 0.5),
        "left_eye": Keypoint(cx - eye_half, eye_y, 0.5),
        "right_eye": Keypoint(cx + eye_half, eye_y, 0.5),
        "left_ear": Keypoint(cx - ear_half, face_cy, 0.4),
        "right_ear": Keypoint(cx + ear_half, face_cy, 0.4),
        "left_shoulder": Keypoint(cx - shoulder_half, shoulder_y, 0.5),
        "right_shoulder": Keypoint(cx + shoulder_half, shoulder_y, 0.5),
        "left_elbow": Keypoint(cx - arm_offset, elbow_y, 0.4),
        "right_elbow": Keypoint(cx + arm_offset, elbow_y, 0.4),
        "left_wrist": Keypoint(cx - arm_offset, wrist_y, 0.4),
        "right_wrist": Keypoint(cx + arm_offset, wrist_y, 0.4),
        "left_hip": Keypoint(cx - hip_half, hip_y, 0.5),
        "right_hip": Keypoint(cx + hip_half, hip_y, 0.5),
        "left_knee": Keypoint(cx - hip_half, knee_y, 0.4),
        "right_knee": Keypoint(cx + hip_half, knee_y, 0.4),
        "left_ankle": Keypoint(cx - hip_half, ankle_y, 0.4),
        "right_ankle": Keypoint(cx + hip_half, ankle_y, 0.4),
    }

    return PoseResult(keypoints=kp, method="heuristic")
