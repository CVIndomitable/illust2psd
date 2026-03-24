"""S4: Semantic Body Part Segmentation — core step.

Primary: SegFormer human parsing (ATR 18-class) for semantic body part masks.
Fallback: Heuristic keypoint-guided regions.
Optional: SAM2 for refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from loguru import logger
from PIL import Image

from illust2psd.config import PipelineConfig, get_taxonomy, get_z_order_map
from illust2psd.steps.s3_pose import Keypoint, PoseResult
from illust2psd.utils.mask_utils import close_mask, fill_holes, remove_small_components


@dataclass
class SegmentResult:
    """Output of segmentation step."""

    masks: dict[str, np.ndarray]  # part_id -> boolean mask (H, W)
    full_masks: dict[str, np.ndarray]  # Before overlap removal (for inpainting)
    method: str


# ATR label ID → our part_id mapping
# ATR: 0=Background, 1=Hat, 2=Hair, 3=Sunglasses, 4=Upper-clothes, 5=Skirt,
#       6=Pants, 7=Dress, 8=Belt, 9=Left-shoe, 10=Right-shoe, 11=Face,
#       12=Left-leg, 13=Right-leg, 14=Left-arm, 15=Right-arm, 16=Bag, 17=Scarf
_ATR_TO_PART = {
    1: "accessory",      # Hat
    2: "hair",           # Hair (split into front/back later)
    3: "accessory",      # Sunglasses
    4: "body",           # Upper-clothes
    5: "body",           # Skirt → body
    6: "body",           # Pants → body
    7: "body",           # Dress → body
    8: "body",           # Belt → body
    9: "left_leg",       # Left-shoe → merge into left_leg
    10: "right_leg",     # Right-shoe → merge into right_leg
    11: "face_base",     # Face
    12: "left_leg",      # Left-leg
    13: "right_leg",     # Right-leg
    14: "left_arm_front",  # Left-arm
    15: "right_arm_front", # Right-arm
    16: "accessory",     # Bag
    17: "accessory",     # Scarf
}


def segment(
    image: Image.Image,
    fg_mask: np.ndarray,
    pose: PoseResult,
    config: PipelineConfig,
) -> SegmentResult:
    """Segment character into body parts.

    Strategy:
    1. SegFormer human parsing (best for anime)
    2. SAM2 with point prompts (optional refinement)
    3. Heuristic region-based fallback

    Args:
        image: RGBA PIL Image
        fg_mask: Boolean foreground mask
        pose: Pose estimation result with keypoints
        config: Pipeline configuration

    Returns:
        SegmentResult with per-part masks
    """
    backend = config.segmentation_backend

    if backend == "segformer":
        try:
            masks = _segformer_segment(image, fg_mask, pose, config)
            method = "segformer"
            logger.info(f"Segmentation with SegFormer: {len(masks)} parts")
        except Exception as e:
            logger.warning(f"SegFormer failed: {e}, falling back to heuristic")
            masks = _heuristic_segment(image, fg_mask, pose, config)
            method = "heuristic"
    elif backend == "sam2":
        try:
            masks = _sam2_segment(image, fg_mask, pose, config)
            method = "sam2"
            logger.info(f"Segmentation with SAM2: {len(masks)} parts")
        except Exception as e:
            logger.warning(f"SAM2 failed: {e}, falling back to heuristic")
            masks = _heuristic_segment(image, fg_mask, pose, config)
            method = "heuristic"
    else:
        masks = _heuristic_segment(image, fg_mask, pose, config)
        method = "heuristic"

    # Stage B: Refine with foreground mask
    for part_id in masks:
        masks[part_id] = masks[part_id] & fg_mask

    # Clean up masks
    for part_id in list(masks.keys()):
        masks[part_id] = close_mask(masks[part_id], 3)
        masks[part_id] = fill_holes(masks[part_id])
        masks[part_id] = remove_small_components(masks[part_id], config.mask_min_part_pixels)
        if not np.any(masks[part_id]):
            del masks[part_id]

    # Store full masks before overlap removal
    full_masks = {k: v.copy() for k, v in masks.items()}

    # Stage C: Resolve overlaps (higher z_order wins)
    masks = _resolve_overlaps(masks)

    # Stage D: Validate coverage
    _validate_coverage(masks, fg_mask)

    return SegmentResult(masks=masks, full_masks=full_masks, method=method)


# ============================================================
# SegFormer backend
# ============================================================

def _segformer_segment(
    image: Image.Image,
    fg_mask: np.ndarray,
    pose: PoseResult,
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Segment using SegFormer human parsing model (ATR 18-class)."""
    import torch

    from illust2psd.models.model_manager import ModelManager

    processor, model = ModelManager.get().get_segformer()

    rgb = image.convert("RGB")
    h, w = fg_mask.shape

    inputs = processor(images=rgb, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    logits = outputs.logits  # (1, 18, H/4, W/4)
    upsampled = torch.nn.functional.interpolate(
        logits, size=(h, w), mode="bilinear", align_corners=False,
    )
    seg_map = upsampled.argmax(dim=1).squeeze().cpu().numpy()

    # Convert ATR labels to our part taxonomy
    raw_masks: dict[str, np.ndarray] = {}
    for atr_id, part_id in _ATR_TO_PART.items():
        atr_mask = seg_map == atr_id
        if not np.any(atr_mask):
            continue
        if part_id in raw_masks:
            raw_masks[part_id] |= atr_mask
        else:
            raw_masks[part_id] = atr_mask.copy()

    # Split hair into front_hair and back_hair
    if "hair" in raw_masks:
        _split_hair(raw_masks, fg_mask, pose, h, w)

    # Add neck: narrow strip between face and body
    if "face_base" in raw_masks and "body" in raw_masks:
        _add_neck(raw_masks, h, w)

    # Ensure all foreground pixels are assigned
    # Unassigned fg pixels go to nearest part
    covered = np.zeros((h, w), dtype=bool)
    for m in raw_masks.values():
        covered |= m
    uncovered = fg_mask & ~covered
    if np.sum(uncovered) > 0:
        # Large uncovered regions likely belong to body or back_hair
        if "back_hair" not in raw_masks:
            raw_masks["back_hair"] = uncovered
        else:
            raw_masks["back_hair"] |= uncovered

    return raw_masks


def _split_hair(
    masks: dict[str, np.ndarray],
    fg_mask: np.ndarray,
    pose: PoseResult,
    h: int, w: int,
) -> None:
    """Split combined hair mask into front_hair and back_hair.

    Front hair = hair pixels above and overlapping the face region (bangs).
    Back hair = everything else (behind/below head).
    """
    hair = masks.pop("hair")

    # Find face center Y as split reference
    face = masks.get("face_base")
    if face is not None and np.any(face):
        face_rows = np.where(np.any(face, axis=1))[0]
        face_top = face_rows[0]
        face_mid = face_rows[len(face_rows) // 3]  # Upper third of face
    else:
        # Estimate from foreground bounding box
        fg_rows = np.where(np.any(fg_mask, axis=1))[0]
        face_top = fg_rows[0]
        face_mid = fg_rows[0] + (fg_rows[-1] - fg_rows[0]) * 0.12

    # Front hair: hair pixels above the upper-third of face
    front_region = np.zeros((h, w), dtype=bool)
    front_region[:int(face_mid), :] = True
    front_hair = hair & front_region

    # Also include any hair overlapping with face (bangs hanging over forehead)
    if face is not None:
        bangs_overlap = hair & face
        front_hair |= bangs_overlap

    back_hair = hair & ~front_hair

    if np.sum(front_hair) > 50:
        masks["front_hair"] = front_hair
    if np.sum(back_hair) > 50:
        masks["back_hair"] = back_hair


def _add_neck(masks: dict[str, np.ndarray], h: int, w: int) -> None:
    """Create neck region between face bottom and body top."""
    face = masks["face_base"]
    body = masks["body"]

    face_rows = np.where(np.any(face, axis=1))[0]
    body_rows = np.where(np.any(body, axis=1))[0]

    if len(face_rows) == 0 or len(body_rows) == 0:
        return

    face_bottom = face_rows[-1]
    body_top = body_rows[0]

    if body_top <= face_bottom:
        # Face and body overlap — neck is the thin gap
        gap = max(3, int((face_bottom - body_top) * 0.3))
        neck_y_start = face_bottom - gap
        neck_y_end = face_bottom + gap
    else:
        # There's a gap between face and body
        neck_y_start = face_bottom
        neck_y_end = body_top

    # Neck width: narrower than face
    face_cols = np.where(np.any(face, axis=0))[0]
    if len(face_cols) == 0:
        return
    face_cx = (face_cols[0] + face_cols[-1]) // 2
    face_w = face_cols[-1] - face_cols[0]
    neck_half_w = max(5, face_w // 4)

    neck = np.zeros((h, w), dtype=bool)
    y1 = max(0, neck_y_start)
    y2 = min(h, neck_y_end)
    x1 = max(0, face_cx - neck_half_w)
    x2 = min(w, face_cx + neck_half_w)
    neck[y1:y2, x1:x2] = True

    # Only keep neck pixels that are in the foreground and not in face/body
    fg_here = face | body
    neck &= ~face & ~body

    if np.sum(neck) > 20:
        masks["neck"] = neck


# ============================================================
# SAM2 backend (kept for optional refinement)
# ============================================================

def _sam2_segment(
    image: Image.Image,
    fg_mask: np.ndarray,
    pose: PoseResult,
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Segment using SAM2 — use SegFormer as coarse, SAM2 for refinement."""
    # First get coarse segmentation from SegFormer
    try:
        coarse = _segformer_segment(image, fg_mask, pose, config)
    except Exception:
        coarse = _heuristic_segment(image, fg_mask, pose, config)

    # Then refine each part with SAM2 using centroid prompts
    from illust2psd.models.model_manager import ModelManager

    predictor = ModelManager.get().get_sam2_predictor(config.device)

    arr = np.array(image)[:, :, :3]
    predictor.set_image(arr)
    h, w = arr.shape[:2]

    refined = {}
    for part_id, mask in coarse.items():
        if np.sum(mask) < 50:
            refined[part_id] = mask
            continue

        # Get centroid of coarse mask
        ys, xs = np.where(mask)
        cx, cy = np.mean(xs), np.mean(ys)

        point_coords = np.array([[cx, cy]], dtype=np.float32)
        point_labels = np.array([1], dtype=np.int32)

        try:
            pred_masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
            # Pick the mask most similar to our coarse mask (highest IoU)
            best_iou = -1
            best_mask = mask
            for i in range(len(scores)):
                candidate = pred_masks[i].astype(bool)
                intersection = np.sum(candidate & mask)
                union = np.sum(candidate | mask)
                iou = intersection / max(union, 1)
                if iou > best_iou:
                    best_iou = iou
                    best_mask = candidate
            refined[part_id] = best_mask
        except Exception:
            refined[part_id] = mask

    return refined


# ============================================================
# Heuristic backend
# ============================================================

def _heuristic_segment(
    image: Image.Image,
    fg_mask: np.ndarray,
    pose: PoseResult,
    config: PipelineConfig,
) -> dict[str, np.ndarray]:
    """Segment using keypoint-guided bounding boxes."""
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

    for prefix in ["left", "right"]:
        shoulder = pose.get(f"{prefix}_shoulder")
        elbow = pose.get(f"{prefix}_elbow")
        wrist = pose.get(f"{prefix}_wrist")
        if shoulder and elbow:
            masks[f"{prefix}_arm_back"] = _limb_mask(h, w, shoulder, elbow, fg_w * 0.08) & fg_mask
        if elbow and wrist:
            masks[f"{prefix}_arm_front"] = _limb_mask(h, w, elbow, wrist, fg_w * 0.07) & fg_mask

    for prefix in ["left", "right"]:
        hip = pose.get(f"{prefix}_hip")
        ankle = pose.get(f"{prefix}_ankle")
        if hip and ankle:
            masks[f"{prefix}_leg"] = _limb_mask(h, w, hip, ankle, fg_w * 0.10) & fg_mask

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


# ============================================================
# Shared utilities
# ============================================================

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
    if not masks:
        return masks

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
    if not masks:
        return

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
