"""CLI entry point for illust2psd."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import click
from loguru import logger

from illust2psd import __version__
from illust2psd.config import PipelineConfig

_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff"}


def _setup_logging(verbose: bool) -> None:
    """Configure loguru logging."""
    logger.remove()
    level = "DEBUG" if verbose else "INFO"
    logger.add(
        sys.stderr, level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )


def _build_config(
    max_size: int = 2048,
    segmentation_backend: str = "heuristic",
    foreground_model: str = "isnet",
    inpaint_backend: str = "opencv",
    device: str = "mps",
    no_inpaint: bool = False,
    dump_masks: str | None = None,
    dump_layers: str | None = None,
    verbose: bool = False,
) -> PipelineConfig:
    return PipelineConfig(
        max_working_size=max_size,
        segmentation_backend=segmentation_backend,
        foreground_model=foreground_model,
        inpaint_backend="none" if no_inpaint else inpaint_backend,
        device=device,
        dump_masks=dump_masks,
        dump_layers=dump_layers,
        verbose=verbose,
    )


@click.group(invoke_without_command=True)
@click.version_option(version=__version__)
@click.pass_context
def main(ctx: click.Context) -> None:
    """illust2psd — Convert anime illustrations to layered PSD for Live2D."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())


@main.command()
@click.argument("input_image", type=click.Path(exists=True))
@click.option("-o", "--output", "output_path", default=None, help="Output PSD path (default: input_name.psd)")
@click.option("--max-size", default=2048, help="Max working size (longest side)")
@click.option("--segmentation-backend", type=click.Choice(["segformer", "sam2", "heuristic"]), default="segformer", help="Segmentation backend")
@click.option("--foreground-model", type=click.Choice(["isnet", "rembg", "grabcut"]), default="isnet", help="Foreground extraction model")
@click.option("--inpaint-backend", type=click.Choice(["lama", "opencv", "none"]), default="opencv", help="Inpainting backend")
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default="mps", help="Compute device")
@click.option("--no-inpaint", is_flag=True, help="Skip inpainting")
@click.option("--dump-masks", type=click.Path(), default=None, help="Save intermediate masks")
@click.option("--dump-layers", type=click.Path(), default=None, help="Save individual layer PNGs")
@click.option("-v", "--verbose", is_flag=True, help="Verbose logging")
def convert(
    input_image: str,
    output_path: str | None,
    max_size: int,
    segmentation_backend: str,
    foreground_model: str,
    inpaint_backend: str,
    device: str,
    no_inpaint: bool,
    dump_masks: str | None,
    dump_layers: str | None,
    verbose: bool,
) -> None:
    """Convert an anime illustration to a layered PSD for Live2D."""
    _setup_logging(verbose)

    if output_path is None:
        output_path = str(Path(input_image).with_suffix(".psd"))

    config = _build_config(
        max_size=max_size, segmentation_backend=segmentation_backend,
        foreground_model=foreground_model, inpaint_backend=inpaint_backend,
        device=device, no_inpaint=no_inpaint, dump_masks=dump_masks,
        dump_layers=dump_layers, verbose=verbose,
    )

    from illust2psd.pipeline import run_pipeline

    t0 = time.time()
    try:
        result = run_pipeline(input_image, output_path, config)
        elapsed = time.time() - t0
        click.echo(f"Done! PSD saved to: {result} ({elapsed:.1f}s)")
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        if verbose:
            raise
        sys.exit(1)


@main.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default=None, help="Output directory (default: input_dir/psd_output)")
@click.option("--max-size", default=2048, help="Max working size")
@click.option("--segmentation-backend", type=click.Choice(["segformer", "sam2", "heuristic"]), default="heuristic")
@click.option("--foreground-model", type=click.Choice(["isnet", "rembg", "grabcut"]), default="isnet")
@click.option("--inpaint-backend", type=click.Choice(["lama", "opencv", "none"]), default="opencv")
@click.option("--device", type=click.Choice(["cuda", "cpu", "mps"]), default="mps")
@click.option("--no-inpaint", is_flag=True)
@click.option("--report", type=click.Path(), default=None, help="Save JSON report to file")
@click.option("-v", "--verbose", is_flag=True)
def batch(
    input_dir: str,
    output_dir: str | None,
    max_size: int,
    segmentation_backend: str,
    foreground_model: str,
    inpaint_backend: str,
    device: str,
    no_inpaint: bool,
    report: str | None,
    verbose: bool,
) -> None:
    """Batch-process all images in a directory."""
    _setup_logging(verbose)

    input_path = Path(input_dir)
    if output_dir is None:
        output_path = input_path / "psd_output"
    else:
        output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    images = sorted(
        f for f in input_path.iterdir()
        if f.is_file() and f.suffix.lower() in _IMAGE_EXTENSIONS
    )

    if not images:
        click.echo(f"No images found in {input_dir}")
        return

    click.echo(f"Found {len(images)} images, output to {output_path}")

    config = _build_config(
        max_size=max_size, segmentation_backend=segmentation_backend,
        foreground_model=foreground_model, inpaint_backend=inpaint_backend,
        device=device, no_inpaint=no_inpaint, verbose=verbose,
    )

    from illust2psd.pipeline import run_pipeline

    results = []
    total_t0 = time.time()

    for i, img_path in enumerate(images, 1):
        psd_path = output_path / f"{img_path.stem}.psd"
        click.echo(f"\n[{i}/{len(images)}] {img_path.name}")

        t0 = time.time()
        try:
            run_pipeline(str(img_path), str(psd_path), config)
            elapsed = time.time() - t0
            size_mb = psd_path.stat().st_size / 1024 / 1024
            results.append({
                "file": img_path.name,
                "status": "success",
                "output": str(psd_path),
                "time_s": round(elapsed, 1),
                "size_mb": round(size_mb, 1),
            })
            click.echo(f"  -> {psd_path.name} ({elapsed:.1f}s, {size_mb:.1f}MB)")
        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "file": img_path.name,
                "status": "error",
                "error": str(e),
                "time_s": round(elapsed, 1),
            })
            click.echo(f"  X Error: {e}")

    total_elapsed = time.time() - total_t0
    success = sum(1 for r in results if r["status"] == "success")

    click.echo(f"\n{'='*50}")
    click.echo(f"Results: {success}/{len(results)} successful ({total_elapsed:.1f}s total)")

    if report:
        report_data = {
            "total": len(results),
            "success": success,
            "total_time_s": round(total_elapsed, 1),
            "results": results,
        }
        Path(report).write_text(json.dumps(report_data, indent=2, ensure_ascii=False))
        click.echo(f"Report saved to: {report}")


@main.command("download-models")
@click.option("--model", type=click.Choice(["isnet_anime", "sam2_hiera_large", "lama", "all"]), default="all")
def download_models(model: str) -> None:
    """Pre-download model weights."""
    _setup_logging(True)

    from illust2psd.utils.download import MODELS, download_model

    if model == "all":
        for name in MODELS:
            click.echo(f"Downloading {name}...")
            download_model(name)
    else:
        download_model(model)

    click.echo("Done!")


@main.command("list-models")
def list_models_cmd() -> None:
    """List available models and cache status."""
    from illust2psd.utils.download import list_models

    models = list_models()
    for name, info in models.items():
        status = "cached" if info["cached"] else "not cached"
        click.echo(f"  {name}: {info['description']} ({info['size_mb']}MB) [{status}]")


if __name__ == "__main__":
    main()
