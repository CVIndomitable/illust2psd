"""S4: Semantic Body Part Segmentation — core step.

Produces per-part binary masks using SAM2 point prompts or heuristic regions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from illust2psd.config import PipelineConfig, get_taxonomy, get_z_order_map
from illust2psd.steps.s3_pose import Keypoint, PoseResult
from illust2psd.utils.mask_utils import remove_small_components


@dataclass
class SegmentResult:
    """Output of segmentation step."""

    masks: dict[str, np.ndarray]  # part_id -> boolean mask (H, W)
    full_masks: dict[str, np.ndarray]  # Before overlap removal (for inpainting)
    method: str


# Maps part_id to keypoint names used to compute center prompt
# and additional negative prompt points (to exclude nearby parts)
_PART_PROMPTS: dict[str, dict] = {
    "face_base": {
        "positive": ["nose", "left_eye", "right_eye"],
        "negative": ["left_shoulder", "right_shoulder"],
        "max_ratio": 0.15,  # Max fraction of image area
    },
    "neck": {
        "positive": ["left_shoulder", "right_shoulder"],
        "offset_y": -0.3,  # Move point slightly above shoulders toward neck
        "negative": ["nose", "left_hip"],
        "max_ratio": 0.05,
    },
    "body": {
        "positive": ["left_shoulder", "right_shoulder", "left_hip", "right_hip"],
        "negative": [],
        "max_ratio": 0.30,
    },
    "left_arm_back": {
        "positive": ["left_shoulder", "left_elbow"],
        "negative": ["right_shoulder"],
        "max_ratio": 0.10,
    },
    "right_arm_back": {
        "positive": ["right_shoulder", "right_elbow"],
        "negative": ["left_shoulder"],
        "max_ratio": 0.10,
    },
    "left_arm_front": {
        "positive": ["left_elbow", "left_wrist"],
        "negative": ["left_shoulder"],
        "max_ratio": 0.10,
    },
    "right_arm_front": {
        "positive": ["right_elbow", "right_wrist"],
        "negative": ["right_shoulder"],
        "max_ratio": 0.10,
    },
    "left_leg": {
        "positive": ["left_hip", "left_knee", "left_ankle"],
        "negative": ["right_hip"],
        "max_ratio": 0.15,
    },
    "right_leg": {
        "positive": ["right_hip", "right_knee", "right_ankle"],
        "negative": ["left_hip"],
        "max_ratio": 0.15,
    },
}


def segment(
    image: Image.Image,
    fg_mask: np.ndarray,
    pose: PoseResult,
    config: PipelineConfig,
) -> SegmentResult:
    """Segment character into body parts.

    Strategy:
    1. SAM2 with point prompts from keypoints
    2. Heuristic region-based fallback

    Args:
        image: RGBA PIL Image
        fg_mask: Boolean foreground mask
        pose: Pose estimation result with keypoints
        config: Pipeline configuration

    Returns:
        SegmentResult with per-part masks
    """
    if config.segmentation_backend == "sam2":
        try:
            masks = _sam2_segment(image, fg_mask, pose, config)
            logger.info(f"Segmentation with SAM2: {len(masks)} parts")
        except Exception as e:
            logger.warning(f"SAM2 failed: {e}, falling back to heuristic")
            masks = _heuristic_segment(image, fg_mask, pose, config)
    else:
        masks = _heuristic_segment(image, fg_mask, pose, config)

    # Stage B: Refine with foreground mask
    for part_id in masks:
        masks[part_id] = masks[part_id] & fg_mask

    # Remove small components
    for part_id in masks:
        masks[part_id] = remove_small_components(masks[part_id], config.mask_min_part_pixels)

    # Store full masks before overlap removal
    full_masks = {k: v.copy() for k, v in masks.items()}

    # Stage C: Resolve overlaps (higher z_order wins)
    masks = _resolve_overlaps(masks)

    # Stage D: Validate coverage
    _validate_coverage(masks, fg_mask)

    method = config.segmentation_backend
    return SegmentResult(masks=masks, full_masks=full_masks, method=method)


def _sam2_segment(
    image: Image.Image,
    fg_mask: np.ndarray,
    pose: PoseResult,
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Segment using SAM2 with point prompts from keypoints."""
    from illust2psd.models.model_manager import ModelManager

    predictor = ModelManager.get().get_sam2_predictor(config.device)

    arr = np.array(image)[:, :, :3]
    predictor.set_image(arr)

    h, w = arr.shape[:2]
    total_pixels = h * w
    masks = {}

    # Get masks for keypoint-based parts using positive + negative prompts
    for part_id, prompt_info in _PART_PROMPTS.items():
        pos_names = prompt_info["positive"]
        neg_names = prompt_info.get("negative", [])
        max_ratio = prompt_info.get("max_ratio", 0.20)

        # Compute positive points
        pos_points = []
        for name in pos_names:
            kp = pose.get(name)
            if kp:
                pos_points.append([kp.x, kp.y])

        if not pos_points:
            continue

        # Apply offset if specified
        offset_y = prompt_info.get("offset_y", 0.0)
        if offset_y != 0.0 and len(pos_points) >= 2:
            center_y = np.mean([p[1] for p in pos_points])
            dy = (pos_points[0][1] - pos_points[-1][1]) * offset_y
            for p in pos_points:
                p[1] += dy

        # Compute negative points
        neg_points = []
        for name in neg_names:
            kp = pose.get(name)
            if kp:
                neg_points.append([kp.x, kp.y])

        # Build coordinate and label arrays
        all_points = pos_points + neg_points
        all_labels = [1] * len(pos_points) + [0] * len(neg_points)

        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(all_labels, dtype=np.int32)

        # Clamp to image bounds
        point_coords[:, 0] = np.clip(point_coords[:, 0], 0, w - 1)
        point_coords[:, 1] = np.clip(point_coords[:, 1], 0, h - 1)

        pred_masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )

        # Pick best mask that isn't too large
        sorted_idx = np.argsort(-scores)
        for idx in sorted_idx:
            candidate = pred_masks[idx].astype(bool)
            ratio = np.sum(candidate) / total_pixels
            if ratio <= max_ratio:
                masks[part_id] = candidate
                break
        else:
            # All too large — pick smallest
            sizes = [np.sum(pred_masks[i]) for i in range(len(scores))]
            masks[part_id] = pred_masks[np.argmin(sizes)].astype(bool)

    # Hair: use subtraction approach with SAM assist
    _add_hair_masks_sam2(masks, image, fg_mask, pose, predictor, h, w)

    # Ears
    _add_ear_masks_sam2(masks, pose, predictor, h, w)

    return masks


def _add_hair_masks_sam2(
    masks: dict[str, np.ndarray],
    image: Image.Image,
    fg_mask: np.ndarray,
    pose: PoseResult,
    predictor,
    h: int, w: int,
) -> None:
    """Add hair masks using SAM2 + subtraction strategy.

    Front hair: SAM2 prompt at top-of-head, constrained above eyes.
    Back hair: Everything in foreground not claimed by other parts.
    """
    # Compute head top region for front hair prompt
    face_kp = pose.get("nose")
    eye_l = pose.get("left_eye")
    eye_r = pose.get("right_eye")

    if face_kp is None:
        return

    # Find the top of the foreground (likely top of hair)
    fg_rows = np.any(fg_mask, axis=1)
    if not fg_rows.any():
        return
    fg_top = np.where(fg_rows)[0][0]

    # Front hair prompt: midpoint between head top and eyes
    if eye_l and eye_r:
        eye_y = min(eye_l.y, eye_r.y)
        eye_cx = (eye_l.x + eye_r.x) / 2
    else:
        eye_y = face_kp.y - h * 0.03
        eye_cx = face_kp.x

    hair_prompt_y = (fg_top + eye_y) / 2
    hair_prompt_x = eye_cx

    # SAM2 prompt for front hair
    point_coords = np.array([[hair_prompt_x, hair_prompt_y]], dtype=np.float32)
    point_labels = np.array([1], dtype=np.int32)
    point_coords[:, 0] = np.clip(point_coords[:, 0], 0, w - 1)
    point_coords[:, 1] = np.clip(point_coords[:, 1], 0, h - 1)

    try:
        pred_masks, scores, _ = predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=True,
        )
        best_idx = np.argmax(scores)
        hair_mask = pred_masks[best_idx].astype(bool) & fg_mask

        # Front hair = hair above eyes
        front_region = np.zeros((h, w), dtype=bool)
        front_region[:int(eye_y + h * 0.02), :] = True
        front_hair = hair_mask & front_region

        # Subtract face
        if "face_base" in masks:
            front_hair &= ~masks["face_base"]

        if np.sum(front_hair) > 100:
            masks["front_hair"] = front_hair
    except Exception as e:
        logger.debug(f"SAM2 hair prompt failed: {e}")

    # Back hair: everything unclaimed
    covered = np.zeros((h, w), dtype=bool)
    for pid, m in masks.items():
        covered |= m
    back_hair = fg_mask & ~covered
    if np.sum(back_hair) > 100:
        masks["back_hair"] = back_hair


def _add_ear_masks_sam2(
    masks: dict[str, np.ndarray],
    pose: PoseResult,
    predictor,
    h: int, w: int,
) -> None:
    """Add ear masks using SAM2 point prompts with size constraint."""
    for part_id, kp_name in [("left_ear", "left_ear"), ("right_ear", "right_ear")]:
        kp = pose.get(kp_name)
        if kp is None:
            continue
        point_coords = np.array([[kp.x, kp.y]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)
        point_coords[:, 0] = np.clip(point_coords[:, 0], 0, w - 1)
        point_coords[:, 1] = np.clip(point_coords[:, 1], 0, h - 1)
        try:
            pred_masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            # Pick smallest reasonable mask (ears are small)
            for idx in np.argsort([np.sum(m) for m in pred_masks]):
                mask = pred_masks[idx].astype(bool)
                ratio = np.sum(mask) / (h * w)
                if 0.001 < ratio < 0.03:
                    masks[part_id] = mask
                    break
        except Exception:
            pass


def _compute_center(pose: PoseResult, kp_names: list[str]) -> tuple[float, float] | None:
    """Compute center point from multiple keypoints."""
    xs, ys = [], []
    for name in kp_names:
        kp = pose.get(name)
        if kp:
            xs.append(kp.x)
            ys.append(kp.y)
    if not xs:
        return None
    return (np.mean(xs), np.mean(ys))


def _heuristic_segment(
    image: Image.Image,
    fg_mask: np.ndarray,
    pose: PoseResult,
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Segment using keypoint-guided bounding boxes and region analysis."""
    h, w = fg_mask.shape
    masks = {}

    rows = np.any(fg_mask, axis=1)
    cols = np.any(fg_mask, axis=0)
    if not rows.any():
        return masks

    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    fg_h = y_max - y_min
    fg_w = x_max - x_min
    cx = (x_min + x_max) / 2

    head_bottom = y_min + fg_h * 0.18
    neck_bottom = y_min + fg_h * 0.22
    shoulder_y = y_min + fg_h * 0.22
    hip_y = y_min + fg_h * 0.52
    body_half = fg_w * 0.20

    masks["face_base"] = _rect_mask(h, w, cx - fg_w * 0.12, y_min + fg_h * 0.04,
                                     cx + fg_w * 0.12, head_bottom) & fg_mask
    masks["neck"] = _rect_mask(h, w, cx - fg_w * 0.06, head_bottom,
                                cx + fg_w * 0.06, neck_bottom) & fg_mask
    masks["body"] = _rect_mask(h, w, cx - body_half, shoulder_y,
                                cx + body_half, hip_y) & fg_mask

    # Arms
    for side, prefix in [("left", "left"), ("right", "right")]:
        shoulder = pose.get(f"{prefix}_shoulder")
        elbow = pose.get(f"{prefix}_elbow")
        wrist = pose.get(f"{prefix}_wrist")
        if shoulder and elbow:
            masks[f"{prefix}_arm_back"] = _limb_mask(h, w, shoulder, elbow, fg_w * 0.08) & fg_mask
        if elbow and wrist:
            masks[f"{prefix}_arm_front"] = _limb_mask(h, w, elbow, wrist, fg_w * 0.07) & fg_mask

    # Legs
    for side, prefix in [("left", "left"), ("right", "right")]:
        hip = pose.get(f"{prefix}_hip")
        ankle = pose.get(f"{prefix}_ankle")
        if hip and ankle:
            masks[f"{prefix}_leg"] = _limb_mask(h, w, hip, ankle, fg_w * 0.10) & fg_mask

    # Hair
    eye_y = y_min + fg_h * 0.10
    front_hair = _rect_mask(h, w, x_min, y_min, x_max, eye_y + fg_h * 0.02) & fg_mask
    if "face_base" in masks:
        front_hair &= ~masks["face_base"]
    if np.sum(front_hair) > 100:
        masks["front_hair"] = front_hair

    covered = np.zeros((h, w), dtype=bool)
    for m in masks.values():
        covered |= m
    back_hair = fg_mask & ~covered
    if np.sum(back_hair) > 100:
        masks["back_hair"] = back_hair

    return masks


def _rect_mask(h: int, w: int, x1: float, y1: float, x2: float, y2: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    y1i, y2i = int(max(0, y1)), int(min(h, y2))
    x1i, x2i = int(max(0, x1)), int(min(w, x2))
    mask[y1i:y2i, x1i:x2i] = True
    return mask


def _limb_mask(h: int, w: int, kp_a: Keypoint, kp_b: Keypoint, width: float) -> np.ndarray:
    mask = np.zeros((h, w), dtype=np.uint8)
    pt1 = (int(kp_a.x), int(kp_a.y))
    pt2 = (int(kp_b.x), int(kp_b.y))
    thickness = max(1, int(width))
    cv2.line(mask, pt1, pt2, 255, thickness)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (thickness, thickness))
    mask = cv2.dilate(mask, kernel)
    return mask > 0


def _resolve_overlaps(masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Resolve overlapping masks: highest z_order wins each pixel."""
    z_map = get_z_order_map()
    h, w = next(iter(masks.values())).shape

    pixel_owner = np.full((h, w), -1, dtype=np.int32)
    pixel_z = np.full((h, w), -1, dtype=np.int32)

    part_ids = list(masks.keys())
    for i, pid in enumerate(part_ids):
        z = z_map.get(pid, 0)
        m = masks[pid]
        higher = m & (z > pixel_z)
        pixel_owner[higher] = i
        pixel_z[higher] = z

    resolved = {}
    for i, pid in enumerate(part_ids):
        resolved[pid] = (pixel_owner == i)

    return resolved


def _validate_coverage(masks: dict[str, np.ndarray], fg_mask: np.ndarray) -> None:
    """Check that masks cover most of the foreground."""
    union = np.zeros_like(fg_mask)
    for m in masks.values():
        union |= m

    fg_pixels = np.sum(fg_mask)
    if fg_pixels == 0:
        return

    covered = np.sum(union & fg_mask)
    coverage = covered / fg_pixels

    if coverage < 0.9:
        logger.warning(f"Part masks cover only {coverage:.1%} of foreground (target >= 90%)")
        uncovered = fg_mask & ~union
        if np.any(uncovered):
            _assign_uncovered(masks, uncovered)
    else:
        logger.debug(f"Coverage: {coverage:.1%}")


def _assign_uncovered(masks: dict[str, np.ndarray], uncovered: np.ndarray) -> None:
    """Assign uncovered foreground pixels to the nearest part."""
    from scipy import ndimage as ndi

    h, w = uncovered.shape
    label_map = np.zeros((h, w), dtype=np.int32)
    part_ids = list(masks.keys())

    for i, pid in enumerate(part_ids, 1):
        label_map[masks[pid]] = i

    dist, indices = ndi.distance_transform_edt(label_map == 0, return_indices=True)

    for i, pid in enumerate(part_ids, 1):
        nearest_label = label_map[indices[0][uncovered], indices[1][uncovered]]
        masks[pid] |= uncovered & (nearest_label == i)
