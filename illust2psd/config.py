"""Global configuration, layer taxonomy, and pipeline presets."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
CONFIG_DIR = PROJECT_ROOT / "configs"
CACHE_DIR = Path.home() / ".cache" / "illust2psd"
MODEL_CACHE_DIR = CACHE_DIR / "models"


@dataclass
class PartDef:
    """Definition of a single body part layer."""

    id: str
    cubism_name: str
    z_order: int
    description: str = ""


@dataclass
class PipelineConfig:
    """Full pipeline configuration."""

    # Image processing
    max_working_size: int = 2048
    min_image_size: int = 256
    max_image_size: int = 4096

    # Segmentation
    segmentation_backend: str = "segformer"  # "segformer" | "sam2" | "heuristic"
    foreground_model: str = "isnet"  # "isnet" | "rembg" | "grabcut"
    sam2_checkpoint: str = "sam2_hiera_large"
    pose_backend: str = "mediapipe"  # "mediapipe" | "dwpose" | "heuristic"

    # Face parsing
    face_parser: str = "sam2"  # "bisenet" | "sam2" | "heuristic"
    face_padding_ratio: float = 0.2

    # Inpainting
    inpaint_backend: str = "lama"  # "lama" | "sd" | "opencv" | "none"
    inpaint_expand_px: int = 5

    # Mask post-processing
    mask_close_kernel: int = 5
    mask_feather_radius: float = 1.5
    mask_min_part_pixels: int = 100

    # Export
    include_reference_layer: bool = True
    psd_color_mode: str = "rgb"
    psd_bit_depth: int = 8

    # Device
    device: str = "mps"  # "cuda" | "cpu" | "mps" (default mps for Apple Silicon)
    torch_dtype: str = "float16"

    # Debug
    dump_masks: str | None = None
    dump_layers: str | None = None
    verbose: bool = False


def load_layer_taxonomy(path: Path | None = None) -> list[PartDef]:
    """Load layer taxonomy from YAML config."""
    if path is None:
        path = CONFIG_DIR / "layer_taxonomy.yaml"
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    parts = []
    for p in data["parts"]:
        parts.append(
            PartDef(
                id=p["id"],
                cubism_name=p["cubism_name"],
                z_order=p["z_order"],
                description=p.get("description", ""),
            )
        )
    return parts


# Pre-sorted taxonomy (back to front)
_taxonomy: list[PartDef] | None = None


def get_taxonomy() -> list[PartDef]:
    """Get the layer taxonomy, sorted by z_order ascending (back to front)."""
    global _taxonomy
    if _taxonomy is None:
        _taxonomy = sorted(load_layer_taxonomy(), key=lambda p: p.z_order)
    return _taxonomy


def get_part_def(part_id: str) -> PartDef | None:
    """Get a PartDef by its id."""
    for p in get_taxonomy():
        if p.id == part_id:
            return p
    return None


def get_z_order_map() -> dict[str, int]:
    """Return mapping of part_id -> z_order."""
    return {p.id: p.z_order for p in get_taxonomy()}


def get_cubism_name_map() -> dict[str, str]:
    """Return mapping of part_id -> cubism_name."""
    return {p.id: p.cubism_name for p in get_taxonomy()}
