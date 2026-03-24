"""End-to-end pipeline tests."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from illust2psd.config import PipelineConfig


class TestPreprocess:
    def test_load_and_normalize(self, sample_image_path: Path):
        from illust2psd.steps.s1_preprocess import preprocess

        config = PipelineConfig()
        result = preprocess(sample_image_path, config)

        assert result.image.mode == "RGBA"
        assert result.original_size == (256, 256)
        assert result.working_size == (256, 256)  # Small enough, no resize

    def test_transparent_background_detection(self, transparent_bg_image: Image.Image, tmp_path: Path):
        from illust2psd.steps.s1_preprocess import preprocess

        path = tmp_path / "transparent.png"
        transparent_bg_image.save(path)

        config = PipelineConfig()
        result = preprocess(path, config)
        assert result.has_transparent_bg is True

    def test_rejects_too_small(self, tmp_path: Path):
        from illust2psd.steps.s1_preprocess import preprocess

        small = Image.new("RGBA", (100, 100))
        path = tmp_path / "small.png"
        small.save(path)

        config = PipelineConfig()
        with pytest.raises(ValueError, match="too small"):
            preprocess(path, config)


class TestForeground:
    def test_alpha_extraction(self, transparent_bg_image: Image.Image):
        from illust2psd.steps.s2_foreground import extract_foreground

        config = PipelineConfig()
        result = extract_foreground(transparent_bg_image, has_transparent_bg=True, config=config)

        assert result.method == "alpha"
        assert result.mask.shape == (256, 256)
        assert result.mask.dtype == bool
        assert np.any(result.mask)

    def test_grabcut_fallback(self, sample_rgba_image: Image.Image):
        from illust2psd.steps.s2_foreground import extract_foreground

        config = PipelineConfig(foreground_model="grabcut")
        result = extract_foreground(sample_rgba_image, has_transparent_bg=False, config=config)

        assert result.method == "grabcut"
        assert result.mask.shape == (256, 256)


class TestPose:
    def test_heuristic_pose(self, sample_rgba_image: Image.Image):
        from illust2psd.steps.s3_pose import estimate_pose

        arr = np.array(sample_rgba_image)
        fg_mask = arr[:, :, 3] > 10

        config = PipelineConfig(pose_backend="heuristic")
        result = estimate_pose(sample_rgba_image, fg_mask, config)

        assert result.method == "heuristic"
        assert "nose" in result.keypoints
        assert "left_shoulder" in result.keypoints


class TestSegment:
    def test_heuristic_segment(self, sample_rgba_image: Image.Image):
        from illust2psd.steps.s3_pose import estimate_pose
        from illust2psd.steps.s4_segment import segment

        arr = np.array(sample_rgba_image)
        fg_mask = arr[:, :, 3] > 10

        config = PipelineConfig(pose_backend="heuristic", segmentation_backend="heuristic")
        pose = estimate_pose(sample_rgba_image, fg_mask, config)
        result = segment(sample_rgba_image, fg_mask, pose, config)

        assert len(result.masks) > 0
        assert result.method == "heuristic"


class TestCompose:
    def test_compose_layers(self):
        from illust2psd.steps.s7_compose import compose_layers

        part_images = {
            "body": np.zeros((256, 256, 4), dtype=np.uint8),
        }
        # Add some opaque pixels
        part_images["body"][100:150, 100:150] = [200, 100, 100, 255]

        config = PipelineConfig()
        layers = compose_layers(part_images, (256, 256), config)

        assert len(layers) == 1
        assert layers[0].name == "Body"


class TestExport:
    def test_psd_export(self, tmp_path: Path):
        from illust2psd.steps.s7_compose import LayerSpec
        from illust2psd.steps.s8_export import export_psd

        layer_img = np.zeros((50, 50, 4), dtype=np.uint8)
        layer_img[:, :] = [200, 100, 100, 255]

        layers = [
            LayerSpec(
                name="Body",
                image=layer_img,
                offset_x=10,
                offset_y=20,
                z_order=100,
                part_id="body",
                confidence=0.9,
            )
        ]

        output = tmp_path / "test.psd"
        config = PipelineConfig(include_reference_layer=False)
        result = export_psd(layers, (256, 256), output, config)

        assert result.exists()
        assert result.stat().st_size > 0
