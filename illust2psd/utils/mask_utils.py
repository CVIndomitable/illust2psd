"""Mask manipulation: morphology, refinement, feathering."""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage


def close_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """Morphological close to fill small holes."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    closed = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_CLOSE, kernel)
    return closed > 127


def open_mask(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """Morphological open to remove small noise."""
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask.astype(np.uint8) * 255, cv2.MORPH_OPEN, kernel)
    return opened > 127


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill holes in binary mask."""
    return ndimage.binary_fill_holes(mask)


def remove_small_components(mask: np.ndarray, min_pixels: int = 100) -> np.ndarray:
    """Remove connected components smaller than min_pixels."""
    labeled, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    sizes = ndimage.sum(mask, labeled, range(1, num_features + 1))
    keep = np.zeros_like(mask, dtype=bool)
    for i, size in enumerate(sizes, 1):
        if size >= min_pixels:
            keep |= labeled == i
    return keep


def feather_edges(mask: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    """Apply Gaussian blur to mask edges for soft alpha.

    Uses distance transform for more precise edge feathering:
    - Interior pixels: alpha = 1.0
    - Edge pixels: smooth gradient based on distance to edge
    - Exterior pixels: alpha = 0.0

    Args:
        mask: Boolean mask
        sigma: Controls the feather width (pixels)

    Returns:
        Float mask [0, 1] with feathered edges
    """
    if sigma <= 0:
        return mask.astype(np.float32)

    mask_u8 = mask.astype(np.uint8) * 255

    # Distance from edge (inside the mask)
    dist_inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, 5)
    # Distance from edge (outside the mask)
    dist_outside = cv2.distanceTransform(255 - mask_u8, cv2.DIST_L2, 5)

    # Smooth transition at edges using sigmoid-like function
    # Pixels far inside: 1.0, pixels at edge: 0.5, pixels far outside: 0.0
    feather_width = sigma * 2
    alpha = np.clip((dist_inside - dist_outside + feather_width) / (2 * feather_width), 0, 1)

    return alpha.astype(np.float32)


def smooth_mask_edges(mask: np.ndarray, blur_radius: int = 3) -> np.ndarray:
    """Smooth jagged mask edges using guided filter or bilateral blur.

    Args:
        mask: Boolean mask
        blur_radius: Smoothing radius

    Returns:
        Smoothed boolean mask
    """
    mask_u8 = mask.astype(np.uint8) * 255

    # Use bilateral filter to smooth edges while preserving shape
    smoothed = cv2.bilateralFilter(mask_u8, blur_radius * 2 + 1, 75, 75)

    return smoothed > 127


def dilate_mask(mask: np.ndarray, pixels: int = 5) -> np.ndarray:
    """Dilate mask by given number of pixels."""
    if pixels <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    dilated = cv2.dilate(mask.astype(np.uint8) * 255, kernel)
    return dilated > 127


def erode_mask(mask: np.ndarray, pixels: int = 2) -> np.ndarray:
    """Erode mask by given number of pixels."""
    if pixels <= 0:
        return mask.copy()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (pixels * 2 + 1, pixels * 2 + 1))
    eroded = cv2.erode(mask.astype(np.uint8) * 255, kernel)
    return eroded > 127


def refine_mask_edges(mask: np.ndarray, image_rgb: np.ndarray | None = None) -> np.ndarray:
    """Refine mask edges using contour approximation and optional color guidance.

    Args:
        mask: Boolean mask to refine
        image_rgb: Optional RGB image for color-guided refinement

    Returns:
        Refined boolean mask
    """
    mask_u8 = mask.astype(np.uint8) * 255

    # Find contours and smooth them
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return mask

    # Approximate contours to reduce jaggedness
    refined = np.zeros_like(mask_u8)
    for cnt in contours:
        epsilon = 0.002 * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        cv2.drawContours(refined, [approx], -1, 255, -1)

    return refined > 127


def mask_to_alpha(mask: np.ndarray, feather_sigma: float = 0.0) -> np.ndarray:
    """Convert boolean mask to uint8 alpha channel, optionally feathered."""
    if feather_sigma > 0:
        alpha = feather_edges(mask, feather_sigma)
        return (alpha * 255).astype(np.uint8)
    return mask.astype(np.uint8) * 255


def apply_mask_to_rgba(
    image: np.ndarray,
    mask: np.ndarray,
    feather_sigma: float = 0.0,
) -> np.ndarray:
    """Apply a mask to an RGBA image, setting alpha from mask."""
    result = image.copy()
    alpha = mask_to_alpha(mask, feather_sigma)
    result[:, :, 3] = np.minimum(result[:, :, 3], alpha)
    return result


def compute_mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    """Compute Intersection over Union between two masks."""
    intersection = np.sum(mask_a & mask_b)
    union = np.sum(mask_a | mask_b)
    if union == 0:
        return 0.0
    return intersection / union
