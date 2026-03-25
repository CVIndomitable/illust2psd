"""S7: Layer Composition — Assemble layers with z-ordering, trim, offset, quality validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from loguru import logger

from illust2psd.config import PipelineConfig, get_cubism_name_map, get_z_order_map
from illust2psd.utils.image_utils import composite_layers, trim_transparent


@dataclass
class LayerSpec:
    """A single layer ready for PSD export."""

    name: str  # Cubism layer name, e.g., "Hair_Front"
    image: np.ndarray  # RGBA uint8 array (trimmed H, W, 4)
    offset_x: int  # X offset in original image coordinates
    offset_y: int  # Y offset in original image coordinates
    z_order: int  # Drawing order (0 = back)
    part_id: str  # Internal part id, e.g., "front_hair"
    confidence: float  # Segmentation confidence


@dataclass
class QualityMetrics:
    """Reconstruction quality metrics."""

    psnr: float  # Peak Signal-to-Noise Ratio (dB)
    ssim: float  # Structural Similarity Index
    coverage: float  # Fraction of foreground pixels covered
    layer_count: int
    total_opaque_pixels: int


def compose_layers(
    part_images: dict[str, np.ndarray],
    original_size: tuple[int, int],
    config: PipelineConfig,
    original_image: np.ndarray | None = None,
    fg_mask: np.ndarray | None = None,
) -> list[LayerSpec]:
    """Assemble final layers from per-part RGBA images.

    Args:
        part_images: Dict of part_id -> RGBA uint8 array (full canvas size)
        original_size: (width, height) of the original image
        config: Pipeline configuration
        original_image: Optional RGBA array for quality comparison
        fg_mask: Optional foreground mask for coverage calculation

    Returns:
        List of LayerSpec sorted by z_order (back to front)
    """
    z_map = get_z_order_map()
    name_map = get_cubism_name_map()
    layers = []

    for part_id, rgba in part_images.items():
        if not np.any(rgba[:, :, 3] > 0):
            logger.warning(f"Skipping empty layer: {part_id}")
            continue

        trimmed, offset_x, offset_y = trim_transparent(rgba, padding=2)

        cubism_name = name_map.get(part_id, part_id)
        z_order = z_map.get(part_id, 0)

        opaque_pixels = np.sum(trimmed[:, :, 3] > 0)
        total_pixels = trimmed.shape[0] * trimmed.shape[1]
        confidence = opaque_pixels / max(total_pixels, 1)

        layer = LayerSpec(
            name=cubism_name,
            image=trimmed,
            offset_x=offset_x,
            offset_y=offset_y,
            z_order=z_order,
            part_id=part_id,
            confidence=confidence,
        )
        layers.append(layer)

        logger.debug(
            f"Layer {cubism_name}: {trimmed.shape[1]}x{trimmed.shape[0]} "
            f"at ({offset_x}, {offset_y}), z={z_order}"
        )

    layers.sort(key=lambda l: l.z_order)

    # Deduplicate names
    _deduplicate_names(layers)

    # Quality validation
    metrics = validate_quality(layers, original_size, original_image, fg_mask)
    _log_metrics(metrics)

    logger.info(f"Composed {len(layers)} layers")
    return layers


def validate_quality(
    layers: list[LayerSpec],
    canvas_size: tuple[int, int],
    original_image: np.ndarray | None = None,
    fg_mask: np.ndarray | None = None,
) -> QualityMetrics:
    """Validate reconstruction quality by compositing layers and comparing.

    Args:
        layers: Assembled layers
        canvas_size: (width, height)
        original_image: RGBA array of original for PSNR/SSIM comparison
        fg_mask: Foreground mask for coverage calculation

    Returns:
        QualityMetrics with PSNR, SSIM, coverage
    """
    # Derive canvas dimensions unambiguously.
    # fg_mask is always numpy (H, W), so use it as the authoritative source when
    # available. Fall back to canvas_size which is PIL (width, height) format.
    if fg_mask is not None:
        canvas_h, canvas_w = fg_mask.shape[:2]
    else:
        canvas_w, canvas_h = canvas_size  # PIL (W, H) → explicit names

    total_opaque = sum(int(np.sum(l.image[:, :, 3] > 0)) for l in layers)

    # Reconstruct composite from layers
    layer_data = [(l.image, l.offset_x, l.offset_y) for l in layers]
    composite = composite_layers(layer_data, canvas_w, canvas_h)

    # Sanity check: composite and fg_mask must have the same spatial dimensions
    if fg_mask is not None and composite.shape[:2] != fg_mask.shape[:2]:
        logger.warning(
            f"Shape mismatch: composite {composite.shape[:2]} vs fg_mask {fg_mask.shape[:2]} — "
            "skipping coverage and PSNR calculation"
        )
        return QualityMetrics(
            psnr=0.0, ssim=0.0, coverage=1.0,
            layer_count=len(layers), total_opaque_pixels=total_opaque,
        )

    # Coverage
    coverage = 1.0
    if fg_mask is not None:
        fg_pixels = np.sum(fg_mask)
        if fg_pixels > 0:
            composite_fg = composite[:, :, 3] > 0
            covered = np.sum(composite_fg & fg_mask)
            coverage = float(covered / fg_pixels)

    # PSNR and SSIM
    psnr = 0.0
    ssim = 0.0
    if original_image is not None:
        psnr, ssim = _compute_psnr_ssim(original_image, composite, fg_mask)

    return QualityMetrics(
        psnr=psnr,
        ssim=ssim,
        coverage=coverage,
        layer_count=len(layers),
        total_opaque_pixels=total_opaque,
    )


def _compute_psnr_ssim(
    original: np.ndarray,
    reconstructed: np.ndarray,
    fg_mask: np.ndarray | None = None,
) -> tuple[float, float]:
    """Compute PSNR and SSIM between original and reconstructed images.

    Only compares within the foreground mask if provided.
    """
    # Work with RGB only (ignore alpha for quality comparison)
    orig_rgb = original[:, :, :3].astype(np.float64)
    recon_rgb = reconstructed[:, :, :3].astype(np.float64)

    if fg_mask is not None:
        # Only compare foreground pixels
        mask_3ch = np.stack([fg_mask] * 3, axis=-1)
        orig_rgb = np.where(mask_3ch, orig_rgb, 0)
        recon_rgb = np.where(mask_3ch, recon_rgb, 0)
        num_pixels = int(np.sum(fg_mask)) * 3
    else:
        num_pixels = orig_rgb.size

    if num_pixels == 0:
        return 0.0, 0.0

    # PSNR
    mse = np.sum((orig_rgb - recon_rgb) ** 2) / num_pixels
    if mse == 0:
        psnr = float("inf")
    else:
        psnr = 10 * np.log10(255.0 ** 2 / mse)

    # SSIM (simplified per-channel, then average)
    ssim = _ssim_channel(original[:, :, :3], reconstructed[:, :, :3], fg_mask)

    return float(psnr), float(ssim)


def _ssim_channel(
    img1: np.ndarray,
    img2: np.ndarray,
    mask: np.ndarray | None = None,
) -> float:
    """Compute mean SSIM across RGB channels.

    Uses the standard SSIM formula with 11x11 Gaussian window.
    """
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2

    ssim_vals = []
    for c in range(3):
        ch1 = img1[:, :, c].astype(np.float64)
        ch2 = img2[:, :, c].astype(np.float64)

        # Gaussian-weighted statistics
        mu1 = cv2.GaussianBlur(ch1, (11, 11), 1.5)
        mu2 = cv2.GaussianBlur(ch2, (11, 11), 1.5)

        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2

        sigma1_sq = cv2.GaussianBlur(ch1 ** 2, (11, 11), 1.5) - mu1_sq
        sigma2_sq = cv2.GaussianBlur(ch2 ** 2, (11, 11), 1.5) - mu2_sq
        sigma12 = cv2.GaussianBlur(ch1 * ch2, (11, 11), 1.5) - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
                   ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

        if mask is not None:
            ssim_val = np.mean(ssim_map[mask])
        else:
            ssim_val = np.mean(ssim_map)

        ssim_vals.append(ssim_val)

    return float(np.mean(ssim_vals))


def _deduplicate_names(layers: list[LayerSpec]) -> None:
    """Ensure no duplicate layer names (required by Cubism)."""
    names = [l.name for l in layers]
    dupes = [n for n in set(names) if names.count(n) > 1]
    if not dupes:
        return

    logger.warning(f"Duplicate layer names: {dupes}")
    seen: dict[str, int] = {}
    for layer in layers:
        if layer.name in seen:
            seen[layer.name] += 1
            layer.name = f"{layer.name}_{seen[layer.name]}"
        else:
            seen[layer.name] = 0


def _log_metrics(metrics: QualityMetrics) -> None:
    """Log quality metrics."""
    logger.info(
        f"Quality: {metrics.layer_count} layers, "
        f"{metrics.total_opaque_pixels} opaque px, "
        f"coverage={metrics.coverage:.1%}"
    )
    if metrics.psnr > 0:
        quality = "excellent" if metrics.psnr >= 35 else \
                  "good" if metrics.psnr >= 30 else \
                  "acceptable" if metrics.psnr >= 25 else "poor"
        logger.info(f"Reconstruction: PSNR={metrics.psnr:.1f}dB, SSIM={metrics.ssim:.4f} ({quality})")
        if metrics.psnr < 25:
            logger.warning("Low PSNR — reconstruction quality may be poor")


def dump_layers(
    layers: list[LayerSpec],
    output_dir: str | Path,
) -> None:
    """Save individual layer PNGs for debugging."""
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for i, layer in enumerate(layers):
        filename = f"{i:02d}_{layer.part_id}_{layer.name}.png"
        img = Image.fromarray(layer.image)
        img.save(output_dir / filename)
        logger.debug(f"Saved layer: {filename}")

    logger.info(f"Dumped {len(layers)} layers to {output_dir}")


def dump_masks(
    masks: dict[str, np.ndarray],
    output_dir: str | Path,
) -> None:
    """Save segmentation masks as PNGs for debugging."""
    from PIL import Image

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for part_id, mask in masks.items():
        filename = f"mask_{part_id}.png"
        img = Image.fromarray((mask.astype(np.uint8) * 255))
        img.save(output_dir / filename)

    logger.info(f"Dumped {len(masks)} masks to {output_dir}")
