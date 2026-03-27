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

    # Extract arms and legs from body using keypoints
    _extract_arms(raw_masks, fg_mask, pose, h, w, image=image)
    _extract_legs(raw_masks, fg_mask, pose, h, w)

    # Add neck: narrow strip between face and body
    if "face_base" in raw_masks and "body" in raw_masks:
        _add_neck(raw_masks, fg_mask, h, w)

    # SAM2 seed expansion: expand SegFormer seeds into full masks
    _sam2_expand_seeds(raw_masks, image, fg_mask, h, w, config)

    # Detect weapons/props and move from body to accessory (if enabled)
    if config.weapon_detection != "none":
        _detect_weapons_gdino(raw_masks, image, fg_mask, h, w, config)

    # Recover hair-colored pixels misclassified as body (long hair problem)
    _recover_hair_from_body(raw_masks, image, fg_mask)

    # Assign uncovered foreground pixels intelligently
    _assign_uncovered_smart(raw_masks, fg_mask, h, w)

    return raw_masks


def _sam2_expand_seeds(
    masks: dict[str, np.ndarray],
    image: Image.Image,
    fg_mask: np.ndarray,
    h: int, w: int,
    config: PipelineConfig,
) -> None:
    """Expand SegFormer seed masks using SAM2 point prompts.

    When SegFormer coverage is low, each classified region is used as a "seed".
    SAM2 is prompted with points sampled from the seed to expand it into a
    more complete mask, boosting overall coverage.
    """
    from illust2psd.utils.mask_utils import remove_small_components

    covered = _union_masks(masks) if masks else np.zeros((h, w), dtype=bool)
    fg_count = int(np.sum(fg_mask))
    covered_count = int(np.sum(covered & fg_mask))
    coverage = covered_count / max(1, fg_count)

    # Only activate if coverage is below threshold
    if coverage >= 0.80:
        logger.debug(f"SAM2 expansion skipped: coverage {coverage:.0%} >= 80%")
        return

    logger.info(f"SAM2 seed expansion: coverage {coverage:.0%}, expanding {len(masks)} parts")

    try:
        from illust2psd.models.model_manager import ModelManager

        predictor = ModelManager.get().get_sam2_predictor(config.device)
        img_arr = np.array(image.convert("RGB"))
        predictor.set_image(img_arr)
    except Exception as e:
        logger.warning(f"SAM2 expansion failed to load: {e}")
        return

    uncovered = fg_mask & ~covered

    # For each part, sample points and expand with SAM2
    # Process larger parts first (body, hair) as they benefit most
    parts_by_size = sorted(masks.keys(), key=lambda k: -np.sum(masks[k]))

    for part_id in parts_by_size:
        seed = masks[part_id]
        seed_px = int(np.sum(seed))
        if seed_px < 50:
            continue

        # Sample prompt points from the seed region
        pos_points = _sample_seed_points(seed, n=5)
        if len(pos_points) < 2:
            continue

        # Sample negative points from OTHER parts (helps SAM2 distinguish)
        neg_points = []
        for other_id, other_mask in masks.items():
            if other_id == part_id or not np.any(other_mask):
                continue
            other_pts = _sample_seed_points(other_mask, n=1)
            neg_points.extend(other_pts)
        neg_points = neg_points[:5]  # Cap at 5 negative points

        all_points = pos_points + neg_points
        all_labels = [1] * len(pos_points) + [0] * len(neg_points)

        point_coords = np.array(all_points, dtype=np.float32)
        point_labels = np.array(all_labels, dtype=np.int32)

        try:
            pred_masks, scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )
        except Exception as e:
            logger.debug(f"SAM2 expansion failed for {part_id}: {e}")
            continue

        # Pick the mask that best extends the seed while staying in uncovered+seed area
        best_mask = None
        best_score = -1
        for m, s in zip(pred_masks, scores):
            candidate = m.astype(bool) & fg_mask
            # How much of the candidate overlaps with existing seed?
            seed_overlap = np.sum(candidate & seed) / max(1, seed_px)
            # How much new area does it claim from uncovered?
            new_area = np.sum(candidate & uncovered)
            # How much does it steal from OTHER parts? (bad)
            other_parts_mask = covered & ~seed
            stolen = np.sum(candidate & other_parts_mask)

            # Good expansion: high seed overlap, substantial new area, minimal stealing
            if seed_overlap < 0.5:
                continue  # Must overlap at least 50% of original seed
            if stolen > new_area * 0.3:
                continue  # Don't steal more than 30% of what you gain

            quality = seed_overlap * 0.3 + (new_area / max(1, fg_count)) * 10 - (stolen / max(1, fg_count)) * 20
            if quality > best_score:
                best_score = quality
                best_mask = candidate

        if best_mask is None:
            continue

        # Only take NEW pixels (uncovered), don't steal from other parts
        expansion = best_mask & uncovered
        expansion = remove_small_components(expansion, min_pixels=50)
        exp_count = int(np.sum(expansion))

        if exp_count < 50:
            continue

        masks[part_id] = seed | expansion
        uncovered = uncovered & ~expansion
        covered = covered | expansion

        logger.debug(f"SAM2 expanded {part_id}: +{exp_count} px (seed {seed_px} → {seed_px + exp_count})")

    new_coverage = int(np.sum(covered & fg_mask)) / max(1, fg_count)
    logger.info(f"SAM2 expansion done: coverage {coverage:.0%} → {new_coverage:.0%}")


def _sample_seed_points(mask: np.ndarray, n: int = 5) -> list[list[float]]:
    """Sample n representative points from a boolean mask.

    Returns points as [[x, y], ...] in image coordinates.
    Samples: centroid + points near boundary for better SAM2 coverage.
    """
    ys, xs = np.where(mask)
    if len(ys) < 10:
        return [[float(xs.mean()), float(ys.mean())]] if len(ys) > 0 else []

    points = []

    # 1. Centroid
    cx, cy = float(xs.mean()), float(ys.mean())
    points.append([cx, cy])

    if n <= 1:
        return points

    # 2. Sample from different spatial regions of the mask
    # Divide the mask bounding box into quadrants and sample from each
    y_min, y_max = ys.min(), ys.max()
    x_min, x_max = xs.min(), xs.max()
    y_mid = (y_min + y_max) / 2
    x_mid = (x_min + x_max) / 2

    quadrants = [
        (xs < x_mid) & (ys < y_mid),  # top-left
        (xs >= x_mid) & (ys < y_mid),  # top-right
        (xs < x_mid) & (ys >= y_mid),  # bottom-left
        (xs >= x_mid) & (ys >= y_mid),  # bottom-right
    ]

    for q_filter in quadrants:
        if len(points) >= n:
            break
        q_idx = np.where(q_filter)[0]
        if len(q_idx) == 0:
            continue
        # Pick a random point near the center of this quadrant
        idx = q_idx[len(q_idx) // 2]
        points.append([float(xs[idx]), float(ys[idx])])

    return points[:n]


def _detect_weapons_gdino(
    masks: dict[str, np.ndarray],
    image: Image.Image,
    fg_mask: np.ndarray,
    h: int, w: int,
    config: PipelineConfig | None = None,
) -> None:
    """Use Grounding DINO (+ optional SAM2) to detect and segment weapons/props from body.

    Modes (via config.weapon_detection):
    - "gdino-sam2": GDINO bbox → SAM2 precise mask (best quality)
    - "gdino-bbox": GDINO bbox only, small boxes directly applied (faster, less precise)
    - "none": skip (handled by caller)
    """
    use_sam2 = config is None or config.weapon_detection == "gdino-sam2"
    body = masks.get("body")
    if body is None or np.sum(body) < 500:
        return

    try:
        import torch
        from illust2psd.models.model_manager import ModelManager

        processor, model = ModelManager.get().get_grounding_dino()

        rgb = image.convert("RGB")
        text = "weapon. sword. gun. cannon. shield. lance. spear. axe. rifle. battleship turret. mechanical equipment."

        inputs = processor(images=rgb, text=text, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            threshold=0.35,
            text_threshold=0.35,
            target_sizes=[(h, w)],
        )[0]

        boxes = results["boxes"]
        scores = results["scores"]
        labels = results["labels"]

        if len(boxes) == 0:
            logger.debug("Grounding DINO: no weapons detected")
            return

        # Collect valid detections
        fg_area = int(np.sum(fg_mask))
        body_area = int(np.sum(body))
        max_box_area = fg_area * (0.60 if use_sam2 else 0.20)

        detections = []  # (x1,y1,x2,y2, cx,cy, label, score)
        for box, score, label in zip(boxes, scores, labels):
            x1, y1, x2, y2 = box.int().tolist()
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            box_area = (x2 - x1) * (y2 - y1)
            if box_area > max_box_area:
                logger.debug(f"GDINO: '{label}' score={score:.2f} box=({x1},{y1},{x2},{y2}) — skip (too large)")
                continue
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            detections.append((x1, y1, x2, y2, cx, cy, label, score))
            mode = "SAM2" if use_sam2 else "bbox"
            logger.debug(f"GDINO: '{label}' score={score:.2f} box=({x1},{y1},{x2},{y2}) → {mode}")

        if not detections:
            logger.debug("Grounding DINO: no valid weapon detections")
            return

        img_arr = np.array(rgb)
        weapon_mask = np.zeros((h, w), dtype=bool)

        if use_sam2:
            # SAM2 refinement: precise weapon masks from bbox center points
            mgr = ModelManager.get()
            device = config.device if config else "mps"
            predictor = mgr.get_sam2_predictor(device)
            predictor.set_image(img_arr)

        for x1, y1, x2, y2, cx, cy, label, score in detections:
            if not use_sam2:
                # gdino-bbox mode: use raw bounding box intersected with body
                weapon_mask[y1:y2, x1:x2] = True
                continue
            point_coords = np.array([[cx, cy]], dtype=np.float32)
            point_labels = np.array([1], dtype=np.int32)

            pred_masks, pred_scores, _ = predictor.predict(
                point_coords=point_coords,
                point_labels=point_labels,
                multimask_output=True,
            )

            # Pick the best mask that overlaps mostly with body (not hair/face)
            best_mask = None
            best_body_overlap = 0
            for m, s in zip(pred_masks, pred_scores):
                candidate = m.astype(bool)
                body_overlap = np.sum(candidate & body) / max(1, np.sum(candidate))
                # Prefer masks that are mostly within the body region
                # and not too large (< 40% of body)
                cand_in_body = np.sum(candidate & body)
                if cand_in_body > body_area * 0.40:
                    continue
                if body_overlap > best_body_overlap:
                    best_body_overlap = body_overlap
                    best_mask = candidate

            if best_mask is not None and best_body_overlap > 0.3:
                # Only keep the part that overlaps with body
                refined = best_mask & body
                weapon_mask |= refined
                logger.debug(f"SAM2 weapon '{label}': {int(np.sum(refined))} px (body overlap {best_body_overlap:.0%})")

        if not np.any(weapon_mask):
            return

        # Skin check: if the SAM2 weapon mask is mostly skin, it's a false positive
        ycrcb = cv2.cvtColor(img_arr, cv2.COLOR_RGB2YCrCb)
        weapon_pixels = ycrcb[weapon_mask]
        skin = (
            (weapon_pixels[:, 0] > 100)
            & (weapon_pixels[:, 1] >= 130) & (weapon_pixels[:, 1] <= 180)
            & (weapon_pixels[:, 2] >= 70) & (weapon_pixels[:, 2] <= 135)
        )
        skin_ratio = np.sum(skin) / len(weapon_pixels)
        if skin_ratio > 0.35:
            logger.debug(f"GDINO+SAM2: weapon mask is {skin_ratio:.0%} skin, skipping (false positive)")
            return

        # Safety: don't move more than 40% of body
        count = int(np.sum(weapon_mask))
        if count > body_area * 0.40:
            logger.debug(f"GDINO+SAM2: would move {count}/{body_area} body px ({count/body_area:.0%}), skipping")
            return

        if count < 100:
            return

        masks["body"] = body & ~weapon_mask
        masks.setdefault("accessory", np.zeros((h, w), dtype=bool))
        masks["accessory"] |= weapon_mask

        logger.debug(f"GDINO+SAM2: moved {count} body px → accessory")

    except Exception as e:
        logger.warning(f"GDINO+SAM2 weapon detection failed: {e}, skipping")


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


def _is_skin_colored(img_rgb: np.ndarray, mask: np.ndarray, min_ratio: float = 0.15) -> bool:
    """Check if a masked region contains enough skin-colored pixels.

    Uses a broad anime skin detection in YCrCb space that covers
    light/medium/dark skin tones common in anime art.
    Returns True if at least min_ratio of the masked pixels are skin-colored.
    """
    if not np.any(mask):
        return False
    ycrcb = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2YCrCb)
    pixels = ycrcb[mask]
    # Anime skin: Y > 100 (not too dark), Cr in [130..180], Cb in [70..135]
    skin = (
        (pixels[:, 0] > 100)
        & (pixels[:, 1] >= 130) & (pixels[:, 1] <= 180)
        & (pixels[:, 2] >= 70) & (pixels[:, 2] <= 135)
    )
    ratio = np.sum(skin) / len(pixels)
    return ratio >= min_ratio


def _extract_arms(
    masks: dict[str, np.ndarray],
    fg_mask: np.ndarray,
    pose: PoseResult,
    h: int, w: int,
    image: Image.Image | None = None,
) -> None:
    """Extract arm regions from body mask using pose keypoints.

    When SegFormer classifies clothed arms as "upper-clothes" (body),
    we use shoulder→elbow→wrist keypoints to carve arm corridors out of
    the body mask.

    Strategy for each arm:
    1. If SegFormer already detected it (skin visible), keep it.
    2. Otherwise, draw a thick corridor from shoulder→elbow→wrist.
    3. Intersect with body mask to get candidate arm pixels.
    4. Validate: candidate must contain skin-colored pixels (rejects weapons).
    5. Subtract from body mask.
    6. Split into arm_back (shoulder→elbow) and arm_front (elbow→wrist).
    """
    body = masks.get("body")
    if body is None:
        return

    img_rgb = np.array(image.convert("RGB")) if image is not None else None

    # Get body bounding box for width reference
    body_cols = np.where(np.any(body, axis=0))[0]
    if len(body_cols) == 0:
        return
    body_w = body_cols[-1] - body_cols[0]
    body_cx = (body_cols[0] + body_cols[-1]) / 2

    for side in ["left", "right"]:
        part_front = f"{side}_arm_front"
        part_back = f"{side}_arm_back"

        # Skip if SegFormer already detected this arm with reasonable size
        if part_front in masks and np.sum(masks[part_front]) > 200:
            continue

        shoulder = pose.get(f"{side}_shoulder")
        elbow = pose.get(f"{side}_elbow")
        wrist = pose.get(f"{side}_wrist")

        if not shoulder:
            continue

        # Corridor width: proportional to body width
        corridor_w = max(10, int(body_w * 0.12))

        # Build arm corridor mask
        arm_mask = np.zeros((h, w), dtype=np.uint8)

        if elbow:
            _draw_thick_line(arm_mask, shoulder, elbow, corridor_w)
            if wrist:
                _draw_thick_line(arm_mask, elbow, wrist, int(corridor_w * 0.85))
        elif wrist:
            _draw_thick_line(arm_mask, shoulder, wrist, corridor_w)
        else:
            body_rows = np.where(np.any(body, axis=1))[0]
            if len(body_rows) == 0:
                continue
            body_h = body_rows[-1] - body_rows[0]
            est_elbow_y = shoulder.y + body_h * 0.35
            dx = -body_w * 0.25 if side == "left" else body_w * 0.25
            est_elbow = Keypoint(shoulder.x + dx, est_elbow_y, 0.3)
            _draw_thick_line(arm_mask, shoulder, est_elbow, corridor_w)

        arm_bool = arm_mask > 0
        arm_bool &= body

        # Don't let arm eat too much of the body (max 30% of body pixels)
        arm_px = np.sum(arm_bool)
        body_px = np.sum(body)
        if arm_px > body_px * 0.30:
            arm_mask2 = np.zeros((h, w), dtype=np.uint8)
            narrow_w = max(8, corridor_w // 2)
            if elbow:
                _draw_thick_line(arm_mask2, shoulder, elbow, narrow_w)
                if wrist:
                    _draw_thick_line(arm_mask2, elbow, wrist, narrow_w)
            arm_bool = (arm_mask2 > 0) & body

        if np.sum(arm_bool) < 50:
            continue

        # Skin color validation: reject corridors that are all weapon/metal
        if img_rgb is not None and not _is_skin_colored(img_rgb, arm_bool):
            logger.debug(f"Skipped {side} arm extraction: no skin pixels (likely weapon/rigging)")
            continue

        # Split into back (shoulder→elbow) and front (elbow→wrist)
        if elbow:
            upper_mask = np.zeros((h, w), dtype=np.uint8)
            _draw_thick_line(upper_mask, shoulder, elbow, corridor_w + 4)
            upper_region = upper_mask > 0

            arm_back = arm_bool & upper_region
            arm_front = arm_bool & ~upper_region

            if np.sum(arm_back) > 30:
                masks[part_back] = arm_back
            if np.sum(arm_front) > 30:
                masks[part_front] = arm_front

            if np.sum(arm_front) <= 30 and np.sum(arm_back) > 30:
                masks[part_front] = arm_bool
                if part_back in masks:
                    del masks[part_back]
        else:
            masks[part_front] = arm_bool

        # Subtract extracted arm from body
        total_arm = np.zeros((h, w), dtype=bool)
        if part_back in masks:
            total_arm |= masks[part_back]
        if part_front in masks:
            total_arm |= masks[part_front]
        masks["body"] = masks["body"] & ~total_arm

        logger.debug(f"Extracted {side} arm: {int(np.sum(total_arm))} px from body")


def _extract_legs(
    masks: dict[str, np.ndarray],
    fg_mask: np.ndarray,
    pose: PoseResult,
    h: int, w: int,
) -> None:
    """Extract leg regions from body mask using pose keypoints.

    When SegFormer classifies stockings/tights as clothing (body),
    we use hip→knee→ankle keypoints to carve leg regions out.
    Only activates if legs weren't already detected by SegFormer.
    """
    body = masks.get("body")
    if body is None:
        return

    for side in ["left", "right"]:
        part_id = f"{side}_leg"

        # Skip if SegFormer already detected a substantial leg
        # (shoes alone don't count — need at least 2% of body pixels)
        body_px = np.sum(body)
        if part_id in masks and np.sum(masks[part_id]) > body_px * 0.02:
            continue

        hip = pose.get(f"{side}_hip")
        knee = pose.get(f"{side}_knee")
        ankle = pose.get(f"{side}_ankle")

        if not hip:
            continue

        # Get body bbox for width reference
        body_cols = np.where(np.any(body, axis=0))[0]
        if len(body_cols) == 0:
            continue
        body_w = body_cols[-1] - body_cols[0]
        corridor_w = max(10, int(body_w * 0.14))

        leg_mask = np.zeros((h, w), dtype=np.uint8)

        if knee:
            _draw_thick_line(leg_mask, hip, knee, corridor_w)
            if ankle:
                _draw_thick_line(leg_mask, knee, ankle, int(corridor_w * 0.9))
        elif ankle:
            _draw_thick_line(leg_mask, hip, ankle, corridor_w)
        else:
            # Estimate: leg extends straight down from hip
            est_ankle = Keypoint(hip.x, h * 0.95, 0.3)
            _draw_thick_line(leg_mask, hip, est_ankle, corridor_w)

        leg_bool = (leg_mask > 0) & body

        # Legs should be in the lower portion of the body mask
        body_rows = np.where(np.any(body, axis=1))[0]
        if len(body_rows) == 0:
            continue
        body_mid_y = body_rows[len(body_rows) // 2]
        # Only keep leg pixels below the body midpoint
        lower_region = np.zeros((h, w), dtype=bool)
        lower_region[int(body_mid_y):, :] = True
        leg_bool &= lower_region

        # Size check: don't let leg eat too much
        if np.sum(leg_bool) > np.sum(body) * 0.35:
            # Narrow the corridor
            leg_mask2 = np.zeros((h, w), dtype=np.uint8)
            narrow_w = max(8, corridor_w // 2)
            if knee:
                _draw_thick_line(leg_mask2, hip, knee, narrow_w)
                if ankle:
                    _draw_thick_line(leg_mask2, knee, ankle, narrow_w)
            leg_bool = (leg_mask2 > 0) & body & lower_region

        if np.sum(leg_bool) < 50:
            continue

        # Merge with existing leg mask if any
        if part_id in masks:
            masks[part_id] |= leg_bool
        else:
            masks[part_id] = leg_bool

        # Subtract from body
        masks["body"] = masks["body"] & ~leg_bool
        logger.debug(f"Extracted {side} leg: {int(np.sum(leg_bool))} px from body")


def _draw_thick_line(
    canvas: np.ndarray,
    kp_a: Keypoint,
    kp_b: Keypoint,
    thickness: int,
) -> None:
    """Draw a thick line between two keypoints on a uint8 canvas."""
    pt1 = (int(kp_a.x), int(kp_a.y))
    pt2 = (int(kp_b.x), int(kp_b.y))
    cv2.line(canvas, pt1, pt2, 255, max(1, thickness))


def _union_masks(masks: dict[str, np.ndarray]) -> np.ndarray:
    """Union all masks into a single boolean array."""
    result = None
    for m in masks.values():
        if result is None:
            result = m.copy()
        else:
            result |= m
    return result if result is not None else np.zeros((1, 1), dtype=bool)


def _assign_uncovered_smart(
    masks: dict[str, np.ndarray],
    fg_mask: np.ndarray,
    h: int, w: int,
) -> None:
    """Assign uncovered foreground pixels intelligently.

    Strategy:
    - Pixels CLOSE to a classified region (within margin) → nearest part
      (with hair/head spatial constraints as before)
    - Pixels FAR from any classified region → accessory
      (these are likely weapons, rigging, props, or other non-human elements)
    """
    from scipy import ndimage as ndi

    covered = _union_masks(masks) if masks else np.zeros((h, w), dtype=bool)
    uncovered = fg_mask & ~covered

    uncovered_count = int(np.sum(uncovered))
    if uncovered_count == 0:
        return

    if not masks:
        masks["accessory"] = uncovered
        return

    # Compute distance from any classified region
    dist_from_classified = ndi.distance_transform_edt(~covered)

    # Dynamic margin based on coverage ratio:
    # High coverage (>80%): 3% diagonal — trust nearest-neighbor for small gaps
    # Low coverage (<40%): 1% diagonal — be conservative, send more to accessory
    fg_count = int(np.sum(fg_mask))
    covered_count = int(np.sum(covered))
    coverage_ratio = covered_count / max(1, fg_count)

    diag = (h ** 2 + w ** 2) ** 0.5
    if coverage_ratio >= 0.80:
        margin_frac = 0.03
    elif coverage_ratio >= 0.50:
        # Linear interpolation: 0.50→0.015, 0.80→0.03
        margin_frac = 0.015 + (coverage_ratio - 0.50) * (0.015 / 0.30)
    else:
        margin_frac = 0.01  # Very conservative for low coverage

    margin_px = max(15, int(diag * margin_frac))
    logger.debug(f"Coverage: {coverage_ratio:.0%}, margin: {margin_px}px ({margin_frac:.3f}×diag)")

    near_uncovered = uncovered & (dist_from_classified <= margin_px)
    far_uncovered = uncovered & (dist_from_classified > margin_px)

    if coverage_ratio < 0.50:
        logger.warning(
            f"Low SegFormer coverage ({coverage_ratio:.0%}). "
            f"Many pixels will go to accessory. Consider --segmentation-backend sam2."
        )

    # --- Near uncovered: assign to nearest part (with head/hair constraints) ---
    if np.any(near_uncovered):
        head_bottom = h * 0.25
        for pid in ["face_base", "front_hair"]:
            if pid in masks and np.any(masks[pid]):
                rows = np.where(np.any(masks[pid], axis=1))[0]
                if len(rows) > 0:
                    head_bottom = max(head_bottom, rows[-1] + h * 0.05)
        head_bottom = min(head_bottom, h * 0.40)

        head_region = np.zeros((h, w), dtype=bool)
        head_region[:int(head_bottom), :] = True

        hair_parts = {"front_hair", "back_hair"}

        label_map_full = np.zeros((h, w), dtype=np.int32)
        label_map_no_hair = np.zeros((h, w), dtype=np.int32)
        part_ids = list(masks.keys())

        for i, pid in enumerate(part_ids, 1):
            label_map_full[masks[pid]] = i
            if pid not in hair_parts:
                label_map_no_hair[masks[pid]] = i

        nearest_full = np.zeros((h, w), dtype=np.int32)
        unlabeled_full = label_map_full == 0
        if np.any(~unlabeled_full):
            _, idx_full = ndi.distance_transform_edt(unlabeled_full, return_indices=True)
            nearest_full = label_map_full[idx_full[0], idx_full[1]]

        nearest_no_hair = np.zeros((h, w), dtype=np.int32)
        unlabeled_nh = label_map_no_hair == 0
        if np.any(~unlabeled_nh):
            _, idx_nh = ndi.distance_transform_edt(unlabeled_nh, return_indices=True)
            nearest_no_hair = label_map_no_hair[idx_nh[0], idx_nh[1]]

        near_head = near_uncovered & head_region
        near_body = near_uncovered & ~head_region

        for i, pid in enumerate(part_ids, 1):
            head_px = near_head & (nearest_full == i)
            if np.any(head_px):
                masks[pid] |= head_px

            if pid not in hair_parts:
                body_px = near_body & (nearest_no_hair == i)
                if np.any(body_px):
                    masks[pid] |= body_px

    # --- Far uncovered + remaining → accessory ---
    still_uncovered = fg_mask & ~_union_masks(masks)
    for i, pid in enumerate(part_ids, 1):
        if pid not in non_body_parts:
            body_px = near_body & (nearest_no_hair == i)
            if np.any(body_px):
                masks[pid] |= body_px

    # --- Everything else (far body-region) → accessory ---
    still_uncovered = fg_mask & ~_union_masks(masks)

    if np.any(still_uncovered):
        masks.setdefault("accessory", np.zeros((h, w), dtype=bool))
        masks["accessory"] |= still_uncovered

    near_count = int(np.sum(near_uncovered))
    far_count = int(np.sum(far_uncovered))
    logger.debug(
        f"Uncovered: {uncovered_count} px total — "
        f"{near_count} near (→nearest part), {far_count} far (→accessory)"
    )


def _recover_hair_from_body(
    masks: dict[str, np.ndarray],
    image: Image.Image,
    fg_mask: np.ndarray,
) -> None:
    """Recover hair-colored pixels that SegFormer misclassified as body.

    Long hair flowing behind/beside the body is often labeled 'upper-clothes'
    by ATR-trained SegFormer.  This function:
    1. Builds a hair color model (mean/std in LAB) from detected hair pixels.
    2. Finds body pixels whose color matches the hair model.
    3. Keeps only candidates that are spatially near existing hair OR in the
       lateral / below-body region (where body-colored long hair actually lives).
    4. Reclassifies them as back_hair.
    """
    if "body" not in masks:
        return

    hair_mask = np.zeros_like(masks["body"])
    for pid in ("back_hair", "front_hair"):
        if pid in masks:
            hair_mask |= masks[pid]

    if not np.any(hair_mask):
        return

    img_rgb = np.array(image.convert("RGB"))
    img_lab = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2LAB).astype(np.float32)

    hair_pixels_lab = img_lab[hair_mask]
    hair_mean = hair_pixels_lab.mean(axis=0)  # L, A, B
    hair_std = hair_pixels_lab.std(axis=0)

    # Tolerance: 2 sigma with floor AND ceiling.
    # Without a cap, high-variance hair (e.g. light blue with highlights)
    # produces L tolerance >100, matching metallic weapons and everything else.
    tol = np.clip(hair_std * 2.0, a_min=[18.0, 10.0, 10.0], a_max=[50.0, 25.0, 25.0])

    diff = np.abs(img_lab - hair_mean)  # (H, W, 3)
    color_match = np.all(diff < tol, axis=2)

    body = masks["body"]
    candidate = body & color_match

    if not np.any(candidate):
        return

    h, w = body.shape

    # Spatial constraint: ONLY pixels adjacent to existing hair (no lateral/below).
    # The old lateral_or_below heuristic was too loose and captured weapon/rigging
    # that happened to sit at the sides of the body.
    dil_px = min(30, max(h, w) // 20)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dil_px * 2 + 1, dil_px * 2 + 1))
    hair_dilated = cv2.dilate(hair_mask.astype(np.uint8), kernel) > 0

    # Require connected components: only keep candidates that form clusters
    # (isolated small fragments are likely mismatches)
    candidate_connected = candidate & hair_dilated
    if not np.any(candidate_connected):
        return

    # Remove small fragments (< 200 px) to avoid weapon debris
    candidate_connected = remove_small_components(candidate_connected, min_pixels=200)

    # Only use adjacency-based candidates (no lateral/below heuristic)
    to_reclassify = candidate_connected

    count = int(np.sum(to_reclassify))
    if count < 50:
        return

    masks["body"] = body & ~to_reclassify

    if "back_hair" in masks:
        masks["back_hair"] |= to_reclassify
    else:
        masks["back_hair"] = to_reclassify

    logger.debug(f"Hair recovery: moved {count} body px → back_hair (hair color ±{tol.astype(int).tolist()})")


def _add_neck(masks: dict[str, np.ndarray], fg_mask: np.ndarray, h: int, w: int) -> None:
    """Create neck region between face bottom and body top.

    The neck is carved from the foreground (primarily from body) in the
    vertical strip just below the face.  Using 1/3 of face width (previously
    1/4 was too narrow for anime characters).  Pixels are taken from the body
    mask rather than from unclaimed gaps, so the neck always has content even
    when face and body are directly adjacent.
    """
    face = masks["face_base"]
    body = masks["body"]

    face_rows = np.where(np.any(face, axis=1))[0]
    body_rows = np.where(np.any(body, axis=1))[0]

    if len(face_rows) == 0 or len(body_rows) == 0:
        return

    face_bottom = int(face_rows[-1])
    body_top = int(body_rows[0])

    # Neck vertical extent: from face bottom, at least face_h * 0.10 (min 40px)
    face_h_px = face_rows[-1] - face_rows[0]
    min_neck_h = max(40, int(face_h_px * 0.10))
    neck_y_start = max(0, face_bottom - 2)           # tiny overlap with face bottom
    neck_y_end = min(h, face_bottom + min_neck_h)

    # Neck horizontal extent: 1/3 of face width, centered on face
    face_cols = np.where(np.any(face, axis=0))[0]
    if len(face_cols) == 0:
        return
    face_cx = (int(face_cols[0]) + int(face_cols[-1])) // 2
    face_w = int(face_cols[-1]) - int(face_cols[0])
    neck_half_w = max(8, face_w // 3)

    neck_region = np.zeros((h, w), dtype=bool)
    neck_region[neck_y_start:neck_y_end, max(0, face_cx - neck_half_w):min(w, face_cx + neck_half_w)] = True

    # Take all foreground pixels in the neck region that belong to body or are unclaimed
    neck = neck_region & fg_mask & ~face
    if not np.any(neck):
        return

    # Carve neck from body (body loses these pixels)
    body_carved = neck & body
    if np.any(body_carved):
        masks["body"] = body & ~body_carved

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

    _, indices = ndi.distance_transform_edt(label_map == 0, return_indices=True)

    # Build full (H,W) nearest-label map using fancy indexing on (H,W) index arrays.
    # The old code did indices[0][uncovered] which produces a (N,) 1D array, then
    # tried to AND it with the (H,W) uncovered mask — shape mismatch.
    nearest_label_map = label_map[indices[0], indices[1]]  # (H,W)

    for i, pid in enumerate(part_ids, 1):
        masks[pid] |= uncovered & (nearest_label_map == i)
