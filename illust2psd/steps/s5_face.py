"""S5: Face Part Extraction — Fine-grained face parsing for eyes, brows, mouth, nose.

Phase 2: Uses SAM2 sub-prompts within the face region for precise extraction.
Falls back to heuristic keypoint-based approach when SAM2 is unavailable.
"""

from __future__ import annotations

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from illust2psd.config import PipelineConfig
from illust2psd.steps.s3_pose import PoseResult
from illust2psd.utils.mask_utils import remove_small_components


def extract_face_parts(
    image: Image.Image,
    face_mask: np.ndarray,
    pose: PoseResult,
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Extract fine-grained face parts from the face region.

    Extracts: left_eye, right_eye, left_eyebrow, right_eyebrow, nose, mouth

    Args:
        image: RGBA PIL Image
        face_mask: Boolean mask of face region
        pose: Pose estimation result
        config: Pipeline configuration

    Returns:
        Dict of part_id -> boolean mask
    """
    h, w = face_mask.shape
    arr = np.array(image)
    masks = {}

    face_rows = np.any(face_mask, axis=1)
    face_cols = np.any(face_mask, axis=0)
    if not face_rows.any():
        logger.warning("Empty face mask, skipping face part extraction")
        return masks

    fy_min, fy_max = np.where(face_rows)[0][[0, -1]]
    fx_min, fx_max = np.where(face_cols)[0][[0, -1]]
    face_h = fy_max - fy_min
    face_w = fx_max - fx_min

    if face_h < 5 or face_w < 5:
        logger.warning(f"Face mask bbox too small ({face_w}×{face_h}px), skipping face extraction")
        return masks

    # Try SAM2 first if configured (independent of main segmentation backend)
    if config.face_parser == "sam2":
        try:
            masks = _sam2_face_parts(arr, face_mask, pose, config, h, w,
                                     fy_min, fy_max, fx_min, fx_max, face_h, face_w)
            if masks:
                # Patch: if any eye mask is suspiciously small (<0.5% of face area),
                # replace it with the heuristic estimate
                face_area = face_h * face_w
                min_eye_px = max(50, int(face_area * 0.005))
                heuristic = None
                for eye_id in ("left_eye", "right_eye"):
                    if eye_id in masks and np.sum(masks[eye_id]) < min_eye_px:
                        logger.debug(
                            f"SAM2 {eye_id} too small ({np.sum(masks[eye_id])} px < {min_eye_px}), "
                            "using heuristic"
                        )
                        if heuristic is None:
                            heuristic = _heuristic_face_parts(
                                arr, face_mask, pose, h, w,
                                fy_min, fy_max, fx_min, fx_max, face_h, face_w, config,
                            )
                        if eye_id in heuristic:
                            masks[eye_id] = heuristic[eye_id]
                logger.info(f"Extracted {len(masks)} face parts with SAM2")
                return masks
        except Exception as e:
            logger.warning(f"SAM2 face extraction failed: {e}, using heuristic")

    # Heuristic fallback
    masks = _heuristic_face_parts(arr, face_mask, pose, h, w,
                                   fy_min, fy_max, fx_min, fx_max,
                                   face_h, face_w, config)

    for pid in masks:
        masks[pid] = remove_small_components(masks[pid], min_pixels=20)

    logger.info(f"Extracted {len(masks)} face parts")
    return masks


def _sam2_face_parts(
    arr: np.ndarray,
    face_mask: np.ndarray,
    pose: PoseResult,
    config: PipelineConfig,
    h: int, w: int,
    fy_min: int, fy_max: int,
    fx_min: int, fx_max: int,
    face_h: int, face_w: int,
) -> dict[str, np.ndarray]:
    """Use SAM2 with face keypoint prompts for precise face part extraction."""
    from illust2psd.models.model_manager import ModelManager

    predictor = ModelManager.get().get_sam2_predictor(config.device)

    # Crop face region with padding for better SAM2 context
    pad = int(max(face_h, face_w) * config.face_padding_ratio)
    crop_y1 = max(0, fy_min - pad)
    crop_y2 = min(h, fy_max + pad)
    crop_x1 = max(0, fx_min - pad)
    crop_x2 = min(w, fx_max + pad)

    if (crop_y2 - crop_y1) < 10 or (crop_x2 - crop_x1) < 10:
        logger.warning(
            f"Face crop too small ({crop_x2 - crop_x1}×{crop_y2 - crop_y1}px), "
            "skipping SAM2 face extraction"
        )
        return {}

    face_crop = arr[crop_y1:crop_y2, crop_x1:crop_x2, :3]
    predictor.set_image(face_crop)

    crop_h, crop_w = face_crop.shape[:2]
    face_cx = (fx_min + fx_max) / 2

    masks = {}

    # Define face part prompts relative to crop coordinates
    part_prompts = _build_face_prompts(pose, face_cx, fy_min, face_h, face_w,
                                        crop_x1, crop_y1, crop_w, crop_h)

    for part_id, prompt in part_prompts.items():
        pos = prompt["positive"]
        neg = prompt.get("negative", [])
        max_ratio = prompt.get("max_ratio", 0.15)

        if not pos:
            continue

        all_points = pos + neg
        all_labels = [1] * len(pos) + [0] * len(neg)

        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(all_labels, dtype=np.int32)
        point_coords[:, 0] = np.clip(point_coords[:, 0], 0, crop_w - 1)
        point_coords[:, 1] = np.clip(point_coords[:, 1], 0, crop_h - 1)

        pred_masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Pick best mask with size constraint
        total_px = crop_h * crop_w
        sorted_idx = np.argsort(-scores)
        for idx in sorted_idx:
            candidate = pred_masks[idx].astype(bool)
            ratio = np.sum(candidate) / total_px
            if ratio <= max_ratio and ratio > 0.001:
                # Map back to full image coordinates
                full_mask = np.zeros((h, w), dtype=bool)
                full_mask[crop_y1:crop_y2, crop_x1:crop_x2] = candidate
                full_mask &= face_mask
                if np.sum(full_mask) > 20:
                    masks[part_id] = full_mask
                break

    return masks


def _build_face_prompts(
    pose: PoseResult,
    face_cx: float,
    fy_min: int,
    face_h: int,
    face_w: int,
    crop_x1: int,
    crop_y1: int,
    crop_w: int,
    crop_h: int,
) -> dict[str, dict]:
    """Build SAM2 prompt points for each face part, in crop coordinates."""
    prompts = {}

    def to_crop(x: float, y: float) -> list[float]:
        return [x - crop_x1, y - crop_y1]

    # Eyes
    l_eye = pose.get("left_eye")
    r_eye = pose.get("right_eye")
    nose_kp = pose.get("nose")

    if l_eye:
        neg = [to_crop(nose_kp.x, nose_kp.y)] if nose_kp else []
        prompts["left_eye"] = {
            "positive": [to_crop(l_eye.x, l_eye.y)],
            "negative": neg,
            "max_ratio": 0.18,
        }
    if r_eye:
        neg = [to_crop(nose_kp.x, nose_kp.y)] if nose_kp else []
        prompts["right_eye"] = {
            "positive": [to_crop(r_eye.x, r_eye.y)],
            "negative": neg,
            "max_ratio": 0.18,
        }

    # Eyebrows — slightly above eyes
    if l_eye:
        brow_y = l_eye.y - face_h * 0.08
        prompts["left_eyebrow"] = {
            "positive": [to_crop(l_eye.x, brow_y)],
            "negative": [to_crop(l_eye.x, l_eye.y)],  # Negative at eye to separate
            "max_ratio": 0.06,
        }
    if r_eye:
        brow_y = r_eye.y - face_h * 0.08
        prompts["right_eyebrow"] = {
            "positive": [to_crop(r_eye.x, brow_y)],
            "negative": [to_crop(r_eye.x, r_eye.y)],
            "max_ratio": 0.06,
        }

    # Nose
    if nose_kp:
        neg = []
        if l_eye:
            neg.append(to_crop(l_eye.x, l_eye.y))
        prompts["nose"] = {
            "positive": [to_crop(nose_kp.x, nose_kp.y)],
            "negative": neg,
            "max_ratio": 0.08,
        }

    # Mouth — below nose
    if nose_kp:
        mouth_y = nose_kp.y + face_h * 0.12
        prompts["mouth"] = {
            "positive": [to_crop(nose_kp.x, mouth_y)],
            "negative": [to_crop(nose_kp.x, nose_kp.y)],
            "max_ratio": 0.08,
        }
    else:
        mouth_y = fy_min + face_h * 0.75
        prompts["mouth"] = {
            "positive": [to_crop(face_cx, mouth_y)],
            "negative": [],
            "max_ratio": 0.08,
        }

    return prompts


def _heuristic_face_parts(
    arr: np.ndarray,
    face_mask: np.ndarray,
    pose: PoseResult,
    h: int, w: int,
    fy_min: int, fy_max: int,
    fx_min: int, fx_max: int,
    face_h: int, face_w: int,
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Extract face parts using keypoint-guided elliptical regions."""
    masks = {}
    face_cx = (fx_min + fx_max) / 2

    eye_w = face_w * 0.18
    eye_h = face_h * 0.12

    # Left eye
    l_eye = pose.get("left_eye")
    if l_eye:
        masks["left_eye"] = _ellipse_mask(h, w, l_eye.x, l_eye.y, eye_w, eye_h) & face_mask
    else:
        ey = fy_min + face_h * 0.40
        ex = face_cx - face_w * 0.15
        masks["left_eye"] = _ellipse_mask(h, w, ex, ey, eye_w, eye_h) & face_mask

    # Right eye
    r_eye = pose.get("right_eye")
    if r_eye:
        masks["right_eye"] = _ellipse_mask(h, w, r_eye.x, r_eye.y, eye_w, eye_h) & face_mask
    else:
        ey = fy_min + face_h * 0.40
        ex = face_cx + face_w * 0.15
        masks["right_eye"] = _ellipse_mask(h, w, ex, ey, eye_w, eye_h) & face_mask

    # Eyebrows
    brow_w = eye_w * 1.3
    brow_h = eye_h * 0.4
    brow_offset_y = eye_h * 1.2

    for eye_id, brow_id in [("left_eye", "left_eyebrow"), ("right_eye", "right_eyebrow")]:
        if eye_id in masks and np.any(masks[eye_id]):
            rows = np.any(masks[eye_id], axis=1)
            ey_top = np.where(rows)[0][0]
            cols = np.any(masks[eye_id], axis=0)
            ex_center = np.mean(np.where(cols)[0])
            masks[brow_id] = _ellipse_mask(h, w, ex_center, ey_top - brow_offset_y,
                                            brow_w, brow_h) & face_mask

    # Nose
    nose_kp = pose.get("nose")
    nose_w = face_w * 0.08
    nose_h = face_h * 0.10
    if nose_kp:
        masks["nose"] = _ellipse_mask(h, w, nose_kp.x, nose_kp.y, nose_w, nose_h) & face_mask
    else:
        masks["nose"] = _ellipse_mask(h, w, face_cx, fy_min + face_h * 0.60,
                                       nose_w, nose_h) & face_mask

    # Mouth
    mouth_w = face_w * 0.15
    mouth_h = face_h * 0.08
    if nose_kp:
        my = nose_kp.y + face_h * 0.12
        mx = nose_kp.x
    else:
        mx = face_cx
        my = fy_min + face_h * 0.75
    masks["mouth"] = _ellipse_mask(h, w, mx, my, mouth_w, mouth_h) & face_mask

    return masks


def _ellipse_mask(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(mask, (int(cx), int(cy)), (int(rx), int(ry)), 0, 0, 360, 255, -1)
    return mask > 0
