"""Image manipulation helpers: crop, pad, resize, alpha compositing."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def ensure_rgba(img: Image.Image) -> Image.Image:
    """Convert image to RGBA mode."""
    if img.mode == "RGBA":
        return img
    return img.convert("RGBA")


def auto_orient(img: Image.Image) -> Image.Image:
    """Apply EXIF orientation and strip EXIF data."""
    return ImageOps.exif_transpose(img)


def resize_to_max(img: Image.Image, max_size: int) -> Image.Image:
    """Resize image so longest side is at most max_size, preserving aspect ratio."""
    w, h = img.size
    if max(w, h) <= max_size:
        return img
    scale = max_size / max(w, h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    return img.resize((new_w, new_h), Image.LANCZOS)


def has_transparent_background(img: Image.Image, threshold: int = 10) -> bool:
    """Check if the image already has a transparent background.

    Returns True if >20% of edge pixels have alpha < threshold.
    """
    if img.mode != "RGBA":
        return False
    arr = np.array(img)
    alpha = arr[:, :, 3]
    h, w = alpha.shape
    # Sample border pixels
    border = np.concatenate([
        alpha[0, :],       # top row
        alpha[-1, :],      # bottom row
        alpha[:, 0],       # left col
        alpha[:, -1],      # right col
    ])
    transparent_ratio = np.mean(border < threshold)
    return bool(transparent_ratio > 0.2)


def trim_transparent(rgba: np.ndarray, padding: int = 2) -> tuple[np.ndarray, int, int]:
    """Trim fully transparent rows/columns, returning trimmed image and offset.

    Args:
        rgba: RGBA uint8 array (H, W, 4)
        padding: Extra transparent pixels to keep around edges

    Returns:
        (trimmed_rgba, offset_x, offset_y)
    """
    alpha = rgba[:, :, 3]
    rows = np.any(alpha > 0, axis=1)
    cols = np.any(alpha > 0, axis=0)

    if not rows.any():
        return rgba, 0, 0

    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    # Apply padding
    h, w = rgba.shape[:2]
    rmin = max(0, rmin - padding)
    rmax = min(h - 1, rmax + padding)
    cmin = max(0, cmin - padding)
    cmax = min(w - 1, cmax + padding)

    trimmed = rgba[rmin : rmax + 1, cmin : cmax + 1]
    return trimmed, cmin, rmin


def composite_layers(
    layers: list[tuple[np.ndarray, int, int]],
    canvas_w: int,
    canvas_h: int,
) -> np.ndarray:
    """Composite multiple RGBA layers onto a canvas (back to front).

    Args:
        layers: List of (rgba_array, offset_x, offset_y) in back-to-front order
        canvas_w: Canvas width
        canvas_h: Canvas height

    Returns:
        Composited RGBA uint8 array
    """
    canvas = np.zeros((canvas_h, canvas_w, 4), dtype=np.uint8)

    for layer, ox, oy in layers:
        lh, lw = layer.shape[:2]
        # Clamp to canvas bounds
        x1 = max(0, ox)
        y1 = max(0, oy)
        x2 = min(canvas_w, ox + lw)
        y2 = min(canvas_h, oy + lh)
        if x2 <= x1 or y2 <= y1:
            continue

        src_x1 = x1 - ox
        src_y1 = y1 - oy
        src_x2 = src_x1 + (x2 - x1)
        src_y2 = src_y1 + (y2 - y1)

        src = layer[src_y1:src_y2, src_x1:src_x2].astype(np.float32)
        dst = canvas[y1:y2, x1:x2].astype(np.float32)

        src_alpha = src[:, :, 3:4] / 255.0
        dst_alpha = dst[:, :, 3:4] / 255.0

        out_alpha = src_alpha + dst_alpha * (1.0 - src_alpha)
        safe_alpha = np.where(out_alpha > 0, out_alpha, 1.0)

        out_rgb = (src[:, :, :3] * src_alpha + dst[:, :, :3] * dst_alpha * (1.0 - src_alpha)) / safe_alpha
        out = np.concatenate([out_rgb, out_alpha * 255.0], axis=2)

        canvas[y1:y2, x1:x2] = np.clip(out, 0, 255).astype(np.uint8)

    return canvas
