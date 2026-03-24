"""PSD file construction helpers using pytoshop."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytoshop
from pytoshop import layers as psd_layers
from pytoshop.enums import ColorMode, Compression


def create_psd(
    canvas_width: int,
    canvas_height: int,
    layer_specs: list[dict],
    output_path: str | Path,
    reference_image: np.ndarray | None = None,
) -> None:
    """Create a PSD file from layer specifications.

    Args:
        canvas_width: PSD canvas width
        canvas_height: PSD canvas height
        layer_specs: List of dicts with keys:
            - name: str (layer name)
            - image: np.ndarray (RGBA uint8, H x W x 4)
            - offset_x: int
            - offset_y: int
        output_path: Where to write the PSD file
        reference_image: Optional flattened reference layer (RGBA)
    """
    psd = pytoshop.PsdFile(num_channels=3, height=canvas_height, width=canvas_width)

    layer_records = []

    # Add reference layer at the bottom (hidden)
    if reference_image is not None:
        ref_layer = _make_layer(
            name="Reference",
            rgba=reference_image,
            offset_x=0,
            offset_y=0,
            visible=False,
        )
        layer_records.append(ref_layer)

    # Add part layers in z-order (back to front = bottom to top in PSD)
    for spec in layer_specs:
        layer = _make_layer(
            name=spec["name"],
            rgba=spec["image"],
            offset_x=spec["offset_x"],
            offset_y=spec["offset_y"],
            visible=True,
        )
        layer_records.append(layer)

    psd.layer_and_mask_info = psd_layers.LayerAndMaskInfo(
        layer_info=psd_layers.LayerInfo(layer_records=layer_records)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        psd.write(f)


def _make_layer(
    name: str,
    rgba: np.ndarray,
    offset_x: int,
    offset_y: int,
    visible: bool = True,
) -> psd_layers.LayerRecord:
    """Create a single PSD layer record from RGBA data."""
    h, w = rgba.shape[:2]

    # Split into channels (contiguous arrays for pytoshop)
    r = np.ascontiguousarray(rgba[:, :, 0])
    g = np.ascontiguousarray(rgba[:, :, 1])
    b = np.ascontiguousarray(rgba[:, :, 2])
    a = np.ascontiguousarray(rgba[:, :, 3])

    # Channel dict: key = channel ID (-1=alpha, 0=R, 1=G, 2=B)
    channels = {
        -1: psd_layers.ChannelImageData(image=a, compression=Compression.raw),
        0: psd_layers.ChannelImageData(image=r, compression=Compression.raw),
        1: psd_layers.ChannelImageData(image=g, compression=Compression.raw),
        2: psd_layers.ChannelImageData(image=b, compression=Compression.raw),
    }

    layer = psd_layers.LayerRecord(
        top=int(offset_y),
        left=int(offset_x),
        bottom=int(offset_y + h),
        right=int(offset_x + w),
        name=name,
        opacity=255,
        visible=visible,
        channels=channels,
    )

    return layer
