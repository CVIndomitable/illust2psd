# CLAUDE.md — Live2D-Ready Layered PSD Generator

## Project Goal

Build a CLI + MCP tool that takes a **single anime/game character illustration (standing picture / 立绘)** and outputs a **high-quality, multi-layered PSD file** ready for import into Live2D Cubism Editor.

The PSD must follow Cubism's layer conventions: one flattened part per layer, RGB color mode, 8-bit channel, sRGB profile, no layer masks, no clipping groups — just clean RGBA layers with correct naming and positioning.

---

## Project Name

`illust2psd`

## Repository Structure

```
illust2psd/
├── CLAUDE.md                  # This file — the implementation guide
├── README.md                  # User-facing documentation
├── pyproject.toml             # Project metadata and dependencies
├── requirements.txt           # Pinned runtime dependencies
│
├── illust2psd/
│   ├── __init__.py
│   ├── cli.py                 # CLI entry point (click-based)
│   ├── pipeline.py            # Orchestrator: chains all steps
│   ├── config.py              # Global constants, layer taxonomy, presets
│   │
│   ├── steps/                 # Each pipeline step is an independent module
│   │   ├── __init__.py
│   │   ├── s1_preprocess.py   # Load image, validate, resize, alpha handling
│   │   ├── s2_foreground.py   # Character foreground extraction (background removal)
│   │   ├── s3_pose.py         # Pose / keypoint estimation
│   │   ├── s4_segment.py      # Semantic body part segmentation (core step)
│   │   ├── s5_face.py         # Fine-grained face part extraction
│   │   ├── s6_inpaint.py      # Occluded region completion
│   │   ├── s7_compose.py      # Layer assembly, z-ordering, trim, offset
│   │   └── s8_export.py       # PSD file generation
│   │
│   ├── models/                # Model loading and inference wrappers
│   │   ├── __init__.py
│   │   ├── seg_model.py       # Segmentation model abstraction
│   │   ├── pose_model.py      # Pose estimation model abstraction
│   │   └── inpaint_model.py   # Inpainting model abstraction
│   │
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── image_utils.py     # Crop, pad, resize, alpha compositing helpers
│   │   ├── mask_utils.py      # Morphological ops, mask refinement, edge feathering
│   │   ├── psd_utils.py       # PSD file construction helpers
│   │   └── download.py        # Model weight downloader with cache
│   │
│   └── mcp_server/            # Optional MCP interface
│       ├── __init__.py
│       └── server.py          # FastMCP server exposing pipeline as tools
│
├── tests/
│   ├── conftest.py
│   ├── test_pipeline.py       # End-to-end tests with sample images
│   ├── test_segmentation.py   # Segmentation quality tests
│   ├── test_psd_export.py     # PSD format compliance tests
│   └── fixtures/              # Small test images
│       └── sample_character.png
│
├── configs/
│   └── layer_taxonomy.yaml    # Layer names, z-order, Cubism naming conventions
│
└── scripts/
    ├── download_models.py     # One-click model weight download
    └── evaluate.py            # Batch evaluation on test set
```

---

## Pipeline Overview

```
Input PNG/JPG
     │
     ▼
┌─────────────────┐
│  S1: Preprocess  │  Validate, normalize to RGBA, resize if needed
└────────┬────────┘
         ▼
┌─────────────────┐
│  S2: Foreground  │  Remove background → binary foreground mask
└────────┬────────┘
         ▼
┌─────────────────┐
│  S3: Pose        │  Detect body keypoints (MediaPipe / DWPose)
└────────┬────────┘
         ▼
┌─────────────────┐
│  S4: Segment     │  Semantic body part segmentation → per-part masks
└────────┬────────┘
         ▼
┌─────────────────┐
│  S5: Face        │  Fine-grained face parsing (eyes, brows, mouth, nose)
└────────┬────────┘
         ▼
┌─────────────────┐
│  S6: Inpaint     │  Complete occluded regions behind each part
└────────┬────────┘
         ▼
┌─────────────────┐
│  S7: Compose     │  Assemble layers: z-order, offset, trim whitespace
└────────┬────────┘
         ▼
┌─────────────────┐
│  S8: Export PSD  │  Write Cubism-compliant layered PSD
└────────┬────────┘
         ▼
    Output .psd
```

---

## Step-by-Step Implementation Details

### S1: Preprocess (`s1_preprocess.py`)

**Input:** Raw image file path (PNG, JPG, WEBP)
**Output:** `PreprocessResult` dataclass with normalized RGBA PIL Image + metadata

Logic:
1. Load image, convert to RGBA
2. Validate dimensions: reject if any side < 256px or > 4096px
3. If image has alpha channel and the background is already transparent, record `has_transparent_bg = True` (skip S2 later)
4. Auto-orient using EXIF
5. Optionally resize to max 2048px on longest side (configurable) for performance. Store original dimensions for final upscale
6. Return `PreprocessResult(image, original_size, working_size, has_transparent_bg)`

### S2: Foreground Extraction (`s2_foreground.py`)

**Input:** RGBA image from S1
**Output:** Binary foreground mask (numpy bool array, same HxW as working image)

Strategy (tiered, try in order):
1. **If `has_transparent_bg`:** derive mask from alpha channel directly (threshold > 10)
2. **Primary: `anime-segmentation` (SkyTNT/anime-segmentation ISNet model)** — best for anime. Use the ONNX exported model (`isnet_is.onnx`) for portability
3. **Fallback: `rembg` with u2net model** — works but less anime-optimized
4. **Emergency fallback: GrabCut** — OpenCV-based, no model needed

Post-processing of mask:
- Morphological close (kernel=5) to fill small holes
- Binary fill holes (`scipy.ndimage.binary_fill_holes`)
- Optional Gaussian blur on edges (sigma=1) for soft alpha

Implementation notes:
- `anime-segmentation` ISNet produces the highest quality masks for anime characters
- Model weights: download to `~/.cache/illust2psd/models/` on first run
- ONNX Runtime inference for cross-platform GPU/CPU support

### S3: Pose Estimation (`s3_pose.py`)

**Input:** RGBA image + foreground mask
**Output:** `PoseResult` with 2D keypoints dict and confidence scores

Purpose: Provide keypoint anchors for S4 (segment) to use as SAM prompts or to validate/correct bounding regions.

Strategy:
1. **Primary: MediaPipe Pose** — lightweight, no GPU needed, detects 33 body landmarks
2. **Alternative: DWPose (via ControlNet aux)** — better for anime, needs GPU
3. **Fallback: heuristic proportions** — assume standard anime character proportions based on foreground mask bounding box

Output keypoints needed (minimum):
- `nose`, `left_eye`, `right_eye`, `left_ear`, `right_ear`
- `left_shoulder`, `right_shoulder`
- `left_elbow`, `right_elbow`, `left_wrist`, `right_wrist`
- `left_hip`, `right_hip`
- `left_knee`, `right_knee`, `left_ankle`, `right_ankle`

Each keypoint: `{"x": float, "y": float, "confidence": float}`

If a keypoint is not detected (`confidence < 0.3`), estimate it from other detected keypoints using body proportion priors.

### S4: Semantic Segmentation (`s4_segment.py`)

**This is the core and most critical step.**

**Input:** RGBA image, foreground mask, pose keypoints
**Output:** `SegmentResult` with per-part binary masks dict

#### Layer Taxonomy (19 parts, aligns with See-through paper)

```yaml
# configs/layer_taxonomy.yaml
parts:
  # --- Back layers (drawn first) ---
  - id: back_hair
    cubism_name: "Hair_Back"
    z_order: 0
    description: "Hair behind the head/body"

  - id: body
    cubism_name: "Body"
    z_order: 100
    description: "Torso/main body"

  - id: left_arm_back
    cubism_name: "Arm_L_Back"
    z_order: 150
    description: "Left upper arm (behind torso if applicable)"

  - id: right_arm_back
    cubism_name: "Arm_R_Back"
    z_order: 150

  - id: left_leg
    cubism_name: "Leg_L"
    z_order: 200

  - id: right_leg
    cubism_name: "Leg_R"
    z_order: 200

  # --- Mid layers ---
  - id: neck
    cubism_name: "Neck"
    z_order: 300

  - id: face_base
    cubism_name: "Face"
    z_order: 400
    description: "Face skin area, no features"

  - id: left_ear
    cubism_name: "Ear_L"
    z_order: 410

  - id: right_ear
    cubism_name: "Ear_R"
    z_order: 410

  # --- Face detail layers ---
  - id: left_eye
    cubism_name: "Eye_L"
    z_order: 500

  - id: right_eye
    cubism_name: "Eye_R"
    z_order: 500

  - id: left_eyebrow
    cubism_name: "Brow_L"
    z_order: 550

  - id: right_eyebrow
    cubism_name: "Brow_R"
    z_order: 550

  - id: nose
    cubism_name: "Nose"
    z_order: 520

  - id: mouth
    cubism_name: "Mouth"
    z_order: 530

  # --- Front layers ---
  - id: left_arm_front
    cubism_name: "Arm_L_Front"
    z_order: 600

  - id: right_arm_front
    cubism_name: "Arm_R_Front"
    z_order: 600

  - id: front_hair
    cubism_name: "Hair_Front"
    z_order: 700
    description: "Bangs and hair in front of face"

  # --- Accessories (optional, detected if present) ---
  - id: accessory
    cubism_name: "Accessory"
    z_order: 800
    description: "Hats, ribbons, glasses, etc."
```

#### Segmentation Strategy (multi-stage)

**Stage A: Coarse Segmentation**
Use a **pre-trained anime body part segmentation model**. Options in priority order:

1. **Train/fine-tune a UNet or SegFormer on anime body parts.**
   - Training data source: extract layers from existing Live2D model databases
     - `Eikanya/Live2d-model` on GitHub has thousands of game Live2D models
     - Each model's texture atlas can be rendered per-part using the Live2D SDK
     - This is exactly how the See-through paper bootstrapped their training data
   - Target: 19-class semantic segmentation at 512x512
   - Architecture: SegFormer-B2 or UNet with EfficientNet backbone
   - This is the highest-effort but highest-quality approach

2. **Use SAM2 with point prompts from S3 keypoints** (more practical for v1)
   - For each body part, compute a center point from relevant keypoints
   - E.g., "head" center = midpoint of left_ear and right_ear keypoints
   - E.g., "left_arm" center = midpoint of left_shoulder and left_elbow
   - Feed each point prompt to SAM2, get per-part masks
   - Requires: `segment-anything-2` + checkpoint (~2.4GB for sam2_hiera_large)
   - Works on GPU (CUDA); CPU is very slow

3. **Keypoint-guided region growing** (lightweight fallback)
   - From keypoints, define approximate bounding boxes per part
   - Within each box, use GrabCut or watershed to segment foreground pixels
   - Lowest quality, but no heavy model needed

**Stage B: Refine with foreground mask**
- AND each part mask with the S2 foreground mask to remove background bleed
- Remove small connected components (< 100 pixels)

**Stage C: Resolve overlaps**
- Parts may overlap (e.g., hair over face). This is expected.
- For each pixel, assign to the part with the HIGHEST z-order that claims it
- Store the "full" mask (before overlap removal) for inpainting reference
- Store the "visible" mask (after overlap removal) for the final layer crop

**Stage D: Validate coverage**
- Check that the union of all part masks covers >= 90% of the foreground mask
- Log warnings for any large uncovered foreground regions
- Assign uncovered foreground pixels to the nearest part

Implementation notes:
- All masks are boolean numpy arrays at working resolution
- Store as `Dict[str, np.ndarray]` keyed by part `id`
- Provide `--segmentation-backend` CLI flag: `sam2` | `heuristic`

### S5: Face Part Extraction (`s5_face.py`)

**Input:** RGBA image, face_base mask from S4, pose keypoints (face landmarks)
**Output:** Refined masks for: left_eye, right_eye, left_eyebrow, right_eyebrow, nose, mouth

Strategy:
1. **Primary: Face parsing model (BiSeNet or similar)**
   - Use a pre-trained face parsing network
   - Crop face region first (from face_base mask bounding box, padded 20%)
   - Run face parser on cropped region, get per-class mask
   - Map parser classes to our taxonomy
   - Note: most face parsers are trained on real faces; for anime, consider `hysts/anime-face-detector` or `nagadomi/lbpcascade_animeface`

2. **SAM2 with face keypoint prompts**
   - Use eye, nose, mouth keypoints from pose estimation as point prompts
   - More reliable for anime style where traditional face parsers struggle

3. **Fallback: keypoint-based bounding boxes**
   - Define small regions around each detected face keypoint
   - Use color/gradient analysis within each region to refine

Quality requirements:
- Eyes must be separated into individual layers (critical for Live2D blink animation)
- Eyebrows must be separate from eyes
- Mouth should include both lips
- If closed mouth detected, still create the layer (Live2D needs it for open/close parameter)

### S6: Inpainting (`s6_inpaint.py`)

**Input:** RGBA image, per-part masks (both "full" and "visible" versions)
**Output:** Per-part RGBA images with occluded regions completed

Purpose: When part A occludes part B, removing A reveals a hole in B. This step fills that hole so each layer is complete when viewed independently.

For each part (in z-order, back to front):
1. Identify occluded region: `occluded = full_mask & ~visible_mask`
   - Actually, the occluded region is where OTHER parts with higher z-order overlap THIS part
   - More precisely: `occluded_by_others = foreground_mask & ~visible_mask_of_this_part & region_that_should_belong_to_this_part`
2. If no occlusion, just crop the original image with the part mask → done
3. If occluded area exists, inpaint those pixels

Inpainting strategy (tiered):
1. **Primary: LaMa (Large Mask inpainting)**
   - Fast, high quality, works well for both anime and realistic styles
   - Model: `LaMa` from `advimman/lama` or via `simple-lama-inpainting` package
   - Input: RGB image + binary mask of area to inpaint
   - Runs on GPU; CPU acceptable for small masks

2. **Alternative: Stable Diffusion Inpainting**
   - Better for large occluded areas where LaMa may produce blurry results
   - Model: `runwayml/stable-diffusion-inpainting` via `diffusers`
   - Heavier (~4GB VRAM), but produces more coherent fills
   - Use prompt: "anime character body part, same art style, seamless"

3. **Lightweight fallback: OpenCV Telea / Navier-Stokes inpainting**
   - `cv2.inpaint(image, mask, radius=3, cv2.INPAINT_TELEA)`
   - Fast but low quality; only for thin occlusion strips

Post-inpainting:
- Composite inpainted pixels back into the part's RGBA layer
- Apply soft alpha feathering at mask edges (Gaussian blur on alpha, radius=2px)

### S7: Layer Composition (`s7_compose.py`)

**Input:** Per-part RGBA images from S6, layer taxonomy config
**Output:** `List[LayerSpec]` ready for PSD export

For each part:
1. **Trim:** Remove fully transparent rows/columns from edges
2. **Record offset:** Store (x, y) offset from original image origin
3. **Ensure minimum padding:** Add 2px transparent border (prevents Cubism import edge artifacts)
4. **Verify naming:** Apply Cubism-compatible layer name from taxonomy
5. **Sort by z-order**

Validation checks:
- Reconstruct composite from all layers → compare with original image
- Compute PSNR and SSIM of reconstruction vs. original (within foreground mask)
- PSNR should be >= 30 dB for acceptable quality
- Log a warning if any layer is empty (0 opaque pixels)

Output `LayerSpec` dataclass:
```python
@dataclass
class LayerSpec:
    name: str           # Cubism layer name, e.g., "Hair_Front"
    image: np.ndarray   # RGBA uint8 array (H, W, 4)
    offset_x: int       # X offset in original image coordinates
    offset_y: int       # Y offset in original image coordinates
    z_order: int        # Drawing order (0 = back)
    part_id: str        # Internal part id, e.g., "front_hair"
    confidence: float   # Segmentation confidence for this part
```

### S8: PSD Export (`s8_export.py`)

**Input:** `List[LayerSpec]`, original image dimensions, config
**Output:** `.psd` file on disk

Use the `psd-tools` library to construct the PSD programmatically.

**Cubism Editor PSD Requirements** (from Live2D official documentation):
- Format: PSD (not PSB)
- Color mode: RGB
- Channel: 8-bit per channel
- Color profile: sRGB
- Each part = one layer (no groups needed, but allowed for organization)
- No layer masks, no adjustment layers, no smart objects
- No layers with duplicate names
- Layer blending mode: Normal
- Layer opacity: 100% (use pixel alpha instead)
- Canvas size = original illustration size

Implementation with `psd-tools`:
```python
# NOTE: psd-tools can READ PSDs well but WRITING is limited.
# For writing, use `pytoshop` (lower-level PSD writer) or
# build PSD bytes manually using the PSD binary spec.
# Alternatively, use `ag-psd` (Node.js) via subprocess.
```

**Recommended PSD writing approach:**

Option A: Use `pytoshop` (pure Python PSD writer)
```
pip install pytoshop
```
- Supports creating multi-layer PSD from numpy arrays
- Handles layer offsets (top, left positioning)
- Can set layer names

Option B: Use `ag-psd` (Node.js, higher quality output)
```
npm install ag-psd
```
- Full PSD spec support
- Better compatibility with Cubism Editor
- Requires Node.js subprocess call

**Choose Option A (`pytoshop`) for the initial implementation** to keep the project pure Python. Fall back to Option B if Cubism compatibility issues arise.

Layer order in PSD (top to bottom in Photoshop = front to back visually):
```
Hair_Front      (z=700, topmost in PSD)
Arm_R_Front     (z=600)
Arm_L_Front     (z=600)
Brow_R          (z=550)
Brow_L          (z=550)
Mouth           (z=530)
Nose            (z=520)
Eye_R           (z=500)
Eye_L           (z=500)
Ear_R           (z=410)
Ear_L           (z=410)
Face            (z=400)
Neck            (z=300)
Leg_R           (z=200)
Leg_L           (z=200)
Arm_R_Back      (z=150)
Arm_L_Back      (z=150)
Body            (z=100)
Hair_Back       (z=0, bottommost in PSD)
```

Also include a flattened "Reference" layer at the very bottom (locked, hidden by default) containing the original illustration for alignment reference.

---

## CLI Interface

```bash
# Basic usage
illust2psd input.png -o output.psd

# With options
illust2psd input.png \
  -o output.psd \
  --max-size 2048 \
  --segmentation-backend sam2 \
  --inpaint-backend lama \
  --device cuda \
  --no-inpaint \           # Skip inpainting (faster, but layers will have holes)
  --dump-masks masks_dir/  # Debug: save intermediate masks as PNGs
  --dump-layers layers_dir/ # Debug: save individual layer PNGs
  --verbose
```

Use `click` for CLI framework. Entry point: `illust2psd.cli:main`

---

## MCP Server Interface

Optional. Wrap the pipeline as MCP tools for use with Claude Code or VS Code.

```python
# illust2psd/mcp_server/server.py
from fastmcp import FastMCP

mcp = FastMCP("illust2psd")

@mcp.tool()
async def convert_to_psd(
    image_path: str,
    output_path: str = "output.psd",
    max_size: int = 2048,
    segmentation_backend: str = "sam2",
    inpaint_backend: str = "lama",
    device: str = "cuda",
) -> dict:
    """Convert an anime illustration to a layered PSD for Live2D."""
    ...

@mcp.tool()
async def preview_segmentation(
    image_path: str,
    device: str = "cuda",
) -> dict:
    """Preview segmentation masks without generating PSD."""
    ...
```

---

## Dependencies

### Core (always required)
```
numpy>=1.24
Pillow>=10.0
opencv-python>=4.8
scipy>=1.11
click>=8.0
pyyaml>=6.0
loguru>=0.7
pytoshop>=1.5        # PSD writing
tqdm>=4.65
```

### Segmentation backends (install what you need)
```
# For foreground extraction
onnxruntime-gpu>=1.16   # or onnxruntime for CPU
# anime-segmentation model weights downloaded separately

# For SAM2 segmentation
segment-anything-2>=0.1
torch>=2.0
torchvision>=0.15

# For MediaPipe pose
mediapipe>=0.10
```

### Inpainting backends (install what you need)
```
# LaMa
simple-lama-inpainting>=0.1

# Stable Diffusion (heavier alternative)
diffusers>=0.25
transformers>=4.35
accelerate>=0.25
```

### Dev dependencies
```
pytest>=7.0
pytest-asyncio>=0.21
black>=23.0
ruff>=0.1
mypy>=1.5
```

### pyproject.toml extras
```toml
[project.optional-dependencies]
cpu = ["onnxruntime>=1.16", "mediapipe>=0.10", "simple-lama-inpainting"]
gpu = ["onnxruntime-gpu>=1.16", "torch>=2.0", "torchvision>=0.15", "segment-anything-2", "simple-lama-inpainting"]
full = ["onnxruntime-gpu>=1.16", "torch>=2.0", "torchvision>=0.15", "segment-anything-2", "simple-lama-inpainting", "diffusers>=0.25", "transformers>=4.35", "accelerate>=0.25"]
dev = ["pytest>=7.0", "pytest-asyncio>=0.21", "black>=23.0", "ruff>=0.1", "mypy>=1.5"]
```

---

## Model Weight Management

All model weights are cached in `~/.cache/illust2psd/models/`.

```python
# illust2psd/utils/download.py
MODELS = {
    "isnet_anime": {
        "url": "https://huggingface.co/skytnt/anime-seg/resolve/main/isnet_is.onnx",
        "filename": "isnet_is.onnx",
        "sha256": "...",
        "size_mb": 176,
    },
    "sam2_hiera_large": {
        "url": "https://dl.fbaipublicfiles.com/segment_anything_2/...",
        "filename": "sam2_hiera_large.pt",
        "sha256": "...",
        "size_mb": 2400,
    },
    "lama": {
        "url": "https://huggingface.co/smartywu/big-lama/resolve/main/big-lama.pt",
        "filename": "big-lama.pt",
        "sha256": "...",
        "size_mb": 200,
    },
}
```

On first run, prompt user to confirm download. Provide `illust2psd download-models` CLI command for pre-download.

---

## Configuration Defaults

```python
# illust2psd/config.py
from dataclasses import dataclass

@dataclass
class PipelineConfig:
    # Image processing
    max_working_size: int = 2048        # Longest side during processing
    min_image_size: int = 256
    max_image_size: int = 4096

    # Segmentation
    segmentation_backend: str = "sam2"  # "sam2" | "heuristic"
    foreground_model: str = "isnet"     # "isnet" | "rembg" | "grabcut"
    sam2_checkpoint: str = "sam2_hiera_large"
    pose_backend: str = "mediapipe"     # "mediapipe" | "dwpose" | "heuristic"

    # Face parsing
    face_parser: str = "sam2"           # "bisenet" | "sam2" | "heuristic"
    face_padding_ratio: float = 0.2     # How much to pad face crop

    # Inpainting
    inpaint_backend: str = "lama"       # "lama" | "sd" | "opencv" | "none"
    inpaint_expand_px: int = 5          # Expand inpaint mask for better blending

    # Mask post-processing
    mask_close_kernel: int = 5
    mask_feather_radius: float = 1.5    # Gaussian sigma for edge softening
    min_part_pixels: int = 100          # Discard parts smaller than this

    # Export
    include_reference_layer: bool = True
    psd_color_mode: str = "rgb"         # Always RGB for Cubism
    psd_bit_depth: int = 8              # Always 8-bit for Cubism

    # Device
    device: str = "cuda"                # "cuda" | "cpu"
    torch_dtype: str = "float16"        # "float16" | "float32"
```

---

## Implementation Priority

Build in this order. Each step should be testable independently.

### Phase 1: Minimal Viable Pipeline (v0.1)
1. `s1_preprocess.py` — image loading (straightforward)
2. `s2_foreground.py` — anime-segmentation ISNet for background removal
3. `s8_export.py` — PSD writer with pytoshop (test with single-layer first)
4. `s3_pose.py` — MediaPipe pose estimation
5. `s4_segment.py` — SAM2 with point prompts from keypoints (core effort)
6. `s7_compose.py` — layer assembly
7. Wire up `pipeline.py` and `cli.py`

**Goal:** Input image → multi-layer PSD with crude but real segmentation. Layers have holes where occluded. No inpainting yet. Test by importing into Cubism Editor.

### Phase 2: Inpainting + Face Detail (v0.2)
8. `s6_inpaint.py` — LaMa inpainting
9. `s5_face.py` — face part extraction with SAM2 sub-prompts
10. Improve mask post-processing (feathering, overlap resolution)

**Goal:** PSD layers are complete (no holes), face parts are properly separated.

### Phase 3: Quality + Polish (v0.3)
11. Quality validation in `s7_compose.py` (PSNR/SSIM check)
12. Heuristic fallbacks for each step (so it works without GPU)
13. MCP server
14. Batch processing support
15. `evaluate.py` script for quality benchmarking

---

## Testing Strategy

### Unit tests
- Each step module has its own test file
- Use small fixture images (256x256 anime characters)
- Mock heavy model inference for fast tests

### Integration tests
- `test_pipeline.py`: Full pipeline on 3-5 test images
- Verify PSD can be opened by `psd-tools` and has expected layer count
- Verify layer names match taxonomy
- Verify composite reconstruction PSNR >= 25 dB

### PSD compliance tests
- Open generated PSD with `psd-tools`, verify:
  - Color mode = RGB
  - Bit depth = 8
  - No duplicate layer names
  - All layers have valid dimensions
  - Canvas size matches original

### Manual validation (not automated)
- Import PSD into Cubism Editor 5.x
- Verify all layers appear correctly
- Verify layers can be moved independently without visible seams

---

## Key Design Decisions

1. **Pure Python (no Node.js dependency for core)** — Use `pytoshop` for PSD writing. Only fall back to `ag-psd` (Node.js) if Cubism compatibility requires it.

2. **Lazy model loading** — Models are loaded on first use, not at import time. This keeps `import illust2psd` fast.

3. **Stateless steps** — Each step function takes explicit inputs and returns explicit outputs. No global state. This makes testing and debugging easy.

4. **Backend abstraction** — Each step has a `Backend` enum and strategy pattern. Adding a new segmentation model = implementing one class.

5. **Fail gracefully** — If a step fails, return partial results with warnings rather than crashing. A PSD with 10 layers is better than no PSD.

6. **No training in this repo** — This repo is inference-only. If we later train a custom segmentation model, that goes in a separate `illust2psd-training` repo.

---

## Known Limitations and Workarounds

| Limitation | Workaround |
|---|---|
| SAM2 struggles with anime hair details | Use the foreground mask to constrain SAM2; hair that SAM misses gets assigned to "front_hair" or "back_hair" by subtraction |
| MediaPipe designed for real humans, may fail on anime | Fall back to heuristic proportions based on foreground mask bounding box; head is top 30%, torso is 30-60%, etc. |
| Inpainting may not match art style perfectly | LaMa generally preserves style well; for critical cases, expose manual override to skip inpainting on specific parts |
| Some characters have non-standard poses (sitting, side view) | v1 targets standing front/slightly-turned poses only. Document this limitation. |
| PSD file size can be large for high-res images | Support `--max-output-size` flag to resize final output; default to original resolution |
| Overlapping accessories (glasses, hats) hard to segment | v1 groups all accessories into one layer. Future: support per-accessory layers |

---

## References

- **See-through paper** (arXiv:2602.03749) — State-of-the-art for this exact task. Monitor for code release.
- **Live2D Cubism PSD guide**: https://docs.live2d.com/en/cubism-editor-manual/precautions-for-psd-data/
- **Live2D material separation guide**: https://docs.live2d.com/en/cubism-editor-tutorials/psd/
- **anime-segmentation**: https://github.com/SkyTNT/anime-segmentation
- **SAM2**: https://github.com/facebookresearch/segment-anything-2
- **LaMa**: https://github.com/advimman/lama
- **pytoshop**: https://github.com/mdboom/pytoshop
- **Eikanya/Live2d-model**: https://github.com/Eikanya/Live2d-model (training data source)
- **livadies-collab/Anime-Segmentation**: https://github.com/livadies-collab/Anime-Segmentation (prior art)
