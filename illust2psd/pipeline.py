"""Pipeline orchestrator — chains all processing steps."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from loguru import logger
from PIL import Image

from illust2psd.config import PipelineConfig
from illust2psd.steps.s1_preprocess import PreprocessResult, preprocess
from illust2psd.steps.s2_foreground import ForegroundResult, extract_foreground
from illust2psd.steps.s3_pose import PoseResult, estimate_pose
from illust2psd.steps.s4_segment import SegmentResult, segment
from illust2psd.steps.s5_face import extract_face_parts
from illust2psd.steps.s6_inpaint import inpaint_parts
from illust2psd.steps.s7_compose import LayerSpec, compose_layers, dump_layers, dump_masks
from illust2psd.steps.s8_export import export_psd


def run_pipeline(
    image_path: str | Path,
    output_path: str | Path,
    config: PipelineConfig | None = None,
) -> Path:
    """Run the full illust2psd pipeline.

    Args:
        image_path: Path to input image (PNG, JPG, WEBP)
        output_path: Path for output PSD file
        config: Pipeline configuration (uses defaults if None)

    Returns:
        Path to the generated PSD file
    """
    if config is None:
        config = PipelineConfig()

    logger.info(f"Starting pipeline: {image_path} → {output_path}")

    # S1: Preprocess
    logger.info("Step 1/8: Preprocess")
    prep: PreprocessResult = preprocess(image_path, config)

    # S2: Foreground extraction
    logger.info("Step 2/8: Foreground extraction")
    fg: ForegroundResult = extract_foreground(
        prep.image, prep.has_transparent_bg, config
    )

    # Debug: dump foreground mask
    if config.dump_masks:
        dump_masks({"foreground": fg.mask}, config.dump_masks)

    # S3: Pose estimation
    logger.info("Step 3/8: Pose estimation")
    pose: PoseResult = estimate_pose(prep.image, fg.mask, config)

    # S4: Semantic segmentation
    logger.info("Step 4/8: Semantic segmentation")
    seg: SegmentResult = segment(prep.image, fg.mask, pose, config)

    # S5: Face part extraction
    logger.info("Step 5/8: Face part extraction")
    face_mask = seg.masks.get("face_base")
    if face_mask is not None and np.any(face_mask):
        face_parts = extract_face_parts(prep.image, face_mask, pose, config)
        # Merge face parts into segmentation results
        for part_id, mask in face_parts.items():
            if np.any(mask):
                seg.masks[part_id] = mask
                seg.full_masks[part_id] = mask.copy()
    else:
        logger.warning("No face_base mask found, skipping face extraction")

    # Debug: dump all masks
    if config.dump_masks:
        dump_masks(seg.masks, config.dump_masks)

    # S6: Inpainting
    logger.info("Step 6/8: Inpainting")
    part_images = inpaint_parts(prep.image, seg.masks, seg.full_masks, config)

    # S7: Layer composition + quality validation
    logger.info("Step 7/8: Layer composition")
    original_arr = np.array(prep.image)
    layers: list[LayerSpec] = compose_layers(
        part_images, prep.working_size, config,
        original_image=original_arr, fg_mask=fg.mask,
    )

    # Debug: dump layers
    if config.dump_layers:
        dump_layers(layers, config.dump_layers)

    # S8: PSD export
    logger.info("Step 8/8: PSD export")
    result_path = export_psd(
        layers=layers,
        canvas_size=prep.working_size,
        output_path=output_path,
        config=config,
        original_image=prep.image,
    )

    logger.info(f"Pipeline complete: {result_path}")
    logger.info(f"  Input: {image_path}")
    logger.info(f"  Output: {result_path}")
    logger.info(f"  Layers: {len(layers)}")
    logger.info(f"  Methods: fg={fg.method}, pose={pose.method}, seg={seg.method}")

    # Release models if not keeping for batch
    try:
        from illust2psd.models.model_manager import ModelManager
        ModelManager.get().clear()
    except Exception:
        pass

    return result_path
