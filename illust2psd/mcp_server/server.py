"""MCP server exposing illust2psd pipeline as tools."""

from __future__ import annotations

import json
from pathlib import Path


def create_server():
    """Create and configure the MCP server."""
    from fastmcp import FastMCP

    mcp = FastMCP("illust2psd")

    @mcp.tool()
    async def convert_to_psd(
        image_path: str,
        output_path: str = "output.psd",
        max_size: int = 2048,
        segmentation_backend: str = "heuristic",
        inpaint_backend: str = "opencv",
        device: str = "mps",
    ) -> dict:
        """Convert an anime illustration to a layered PSD for Live2D.

        Args:
            image_path: Path to input image (PNG, JPG, WEBP)
            output_path: Path for output PSD file
            max_size: Max working size (longest side)
            segmentation_backend: "sam2" or "heuristic"
            inpaint_backend: "lama", "opencv", or "none"
            device: "cuda", "cpu", or "mps"
        """
        from illust2psd.config import PipelineConfig
        from illust2psd.pipeline import run_pipeline

        config = PipelineConfig(
            max_working_size=max_size,
            segmentation_backend=segmentation_backend,
            inpaint_backend=inpaint_backend,
            device=device,
        )

        result_path = run_pipeline(image_path, output_path, config)
        size_mb = result_path.stat().st_size / 1024 / 1024

        return {
            "output_path": str(result_path),
            "size_mb": round(size_mb, 1),
            "status": "success",
        }

    @mcp.tool()
    async def preview_segmentation(
        image_path: str,
        output_dir: str = "preview_masks",
        segmentation_backend: str = "heuristic",
        device: str = "mps",
    ) -> dict:
        """Preview segmentation masks without generating PSD.

        Args:
            image_path: Path to input image
            output_dir: Directory to save mask previews
            segmentation_backend: "sam2" or "heuristic"
            device: Compute device
        """
        from illust2psd.config import PipelineConfig
        from illust2psd.steps.s1_preprocess import preprocess
        from illust2psd.steps.s2_foreground import extract_foreground
        from illust2psd.steps.s3_pose import estimate_pose
        from illust2psd.steps.s4_segment import segment
        from illust2psd.steps.s7_compose import dump_masks

        config = PipelineConfig(
            device=device,
            segmentation_backend=segmentation_backend,
        )

        prep = preprocess(image_path, config)
        fg = extract_foreground(prep.image, prep.has_transparent_bg, config)
        pose = estimate_pose(prep.image, fg.mask, config)
        seg = segment(prep.image, fg.mask, pose, config)

        dump_masks(seg.masks, output_dir)

        return {
            "output_dir": output_dir,
            "parts": list(seg.masks.keys()),
            "part_count": len(seg.masks),
            "methods": {
                "foreground": fg.method,
                "pose": pose.method,
                "segmentation": seg.method,
            },
            "status": "success",
        }

    @mcp.tool()
    async def list_models() -> dict:
        """List available models and their cache status."""
        from illust2psd.utils.download import list_models as _list

        models = _list()
        return {
            "models": {
                name: {
                    "description": info["description"],
                    "size_mb": info["size_mb"],
                    "cached": info["cached"],
                }
                for name, info in models.items()
            }
        }

    @mcp.tool()
    async def batch_convert(
        input_dir: str,
        output_dir: str = "psd_output",
        segmentation_backend: str = "heuristic",
        device: str = "mps",
    ) -> dict:
        """Batch-convert all images in a directory to PSD.

        Args:
            input_dir: Directory containing input images
            output_dir: Directory for output PSD files
            segmentation_backend: "sam2" or "heuristic"
            device: Compute device
        """
        from illust2psd.config import PipelineConfig
        from illust2psd.pipeline import run_pipeline

        config = PipelineConfig(
            segmentation_backend=segmentation_backend,
            device=device,
        )

        input_path = Path(input_dir)
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        extensions = {".png", ".jpg", ".jpeg", ".webp"}
        images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in extensions)

        results = []
        for img in images:
            psd = out_path / f"{img.stem}.psd"
            try:
                run_pipeline(str(img), str(psd), config)
                results.append({"file": img.name, "status": "success"})
            except Exception as e:
                results.append({"file": img.name, "status": "error", "error": str(e)})

        success = sum(1 for r in results if r["status"] == "success")
        return {
            "total": len(results),
            "success": success,
            "output_dir": str(out_path),
            "results": results,
        }

    return mcp


if __name__ == "__main__":
    server = create_server()
    server.run()
