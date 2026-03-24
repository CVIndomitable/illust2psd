#!/usr/bin/env python3
"""Batch evaluation script for illust2psd quality benchmarking.

Processes all images, generates PSDs, and reports quality metrics
(PSNR, SSIM, coverage, timing) in a CSV/JSON report.
"""

import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import click
import numpy as np
from loguru import logger


@click.command()
@click.argument("input_dir", type=click.Path(exists=True))
@click.option("-o", "--output-dir", default="eval_output", help="Output directory")
@click.option("--device", default="mps", help="Compute device")
@click.option("--segmentation-backend", default="heuristic", type=click.Choice(["sam2", "heuristic"]))
@click.option("--dump-layers", is_flag=True, help="Also dump individual layers")
@click.option("--report-format", type=click.Choice(["json", "csv", "both"]), default="both")
def evaluate(
    input_dir: str,
    output_dir: str,
    device: str,
    segmentation_backend: str,
    dump_layers: bool,
    report_format: str,
):
    """Batch-process images and report quality metrics."""
    from illust2psd.config import PipelineConfig
    from illust2psd.steps.s1_preprocess import preprocess
    from illust2psd.steps.s2_foreground import extract_foreground
    from illust2psd.steps.s3_pose import estimate_pose
    from illust2psd.steps.s4_segment import segment
    from illust2psd.steps.s5_face import extract_face_parts
    from illust2psd.steps.s6_inpaint import inpaint_parts
    from illust2psd.steps.s7_compose import compose_layers, dump_layers as _dump_layers, validate_quality
    from illust2psd.steps.s8_export import export_psd

    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    extensions = {".png", ".jpg", ".jpeg", ".webp"}
    images = sorted(f for f in input_path.iterdir() if f.suffix.lower() in extensions)

    if not images:
        click.echo(f"No images found in {input_dir}")
        return

    click.echo(f"Evaluating {len(images)} images")

    config = PipelineConfig(
        device=device,
        segmentation_backend=segmentation_backend,
        inpaint_backend="opencv",
        dump_layers=str(output_path / "layers") if dump_layers else None,
    )

    results = []
    total_t0 = time.time()

    for i, img_path in enumerate(images, 1):
        click.echo(f"\n[{i}/{len(images)}] {img_path.name}")
        psd_path = output_path / f"{img_path.stem}.psd"

        t0 = time.time()
        try:
            # Run pipeline steps manually for metrics access
            prep = preprocess(str(img_path), config)
            fg = extract_foreground(prep.image, prep.has_transparent_bg, config)
            pose = estimate_pose(prep.image, fg.mask, config)
            seg = segment(prep.image, fg.mask, pose, config)

            face_mask = seg.masks.get("face_base")
            if face_mask is not None and np.any(face_mask):
                face_parts = extract_face_parts(prep.image, face_mask, pose, config)
                for pid, mask in face_parts.items():
                    if np.any(mask):
                        seg.masks[pid] = mask
                        seg.full_masks[pid] = mask.copy()

            part_images = inpaint_parts(prep.image, seg.masks, seg.full_masks, config)

            original_arr = np.array(prep.image)
            layers = compose_layers(
                part_images, prep.working_size, config,
                original_image=original_arr, fg_mask=fg.mask,
            )

            # Get quality metrics
            metrics = validate_quality(layers, prep.working_size, original_arr, fg.mask)

            export_psd(layers, prep.working_size, psd_path, config, prep.image)

            elapsed = time.time() - t0
            size_mb = psd_path.stat().st_size / 1024 / 1024

            result = {
                "file": img_path.name,
                "status": "success",
                "time_s": round(elapsed, 1),
                "size_mb": round(size_mb, 2),
                "layers": metrics.layer_count,
                "psnr": round(metrics.psnr, 2),
                "ssim": round(metrics.ssim, 4),
                "coverage": round(metrics.coverage, 4),
                "opaque_pixels": metrics.total_opaque_pixels,
                "fg_method": fg.method,
                "pose_method": pose.method,
                "seg_method": seg.method,
            }
            results.append(result)

            quality = "good" if metrics.psnr >= 30 else "acceptable" if metrics.psnr >= 25 else "poor"
            click.echo(
                f"  PSNR={metrics.psnr:.1f}dB SSIM={metrics.ssim:.3f} "
                f"coverage={metrics.coverage:.1%} layers={metrics.layer_count} "
                f"({quality}, {elapsed:.1f}s)"
            )

        except Exception as e:
            elapsed = time.time() - t0
            results.append({
                "file": img_path.name,
                "status": "error",
                "error": str(e),
                "time_s": round(elapsed, 1),
            })
            click.echo(f"  ERROR: {e}")

    total_elapsed = time.time() - total_t0
    success_results = [r for r in results if r["status"] == "success"]

    # Summary
    click.echo(f"\n{'='*60}")
    click.echo(f"Results: {len(success_results)}/{len(results)} successful ({total_elapsed:.1f}s)")

    if success_results:
        avg_psnr = np.mean([r["psnr"] for r in success_results])
        avg_ssim = np.mean([r["ssim"] for r in success_results])
        avg_coverage = np.mean([r["coverage"] for r in success_results])
        avg_layers = np.mean([r["layers"] for r in success_results])
        avg_time = np.mean([r["time_s"] for r in success_results])

        click.echo(f"Avg PSNR:     {avg_psnr:.1f} dB")
        click.echo(f"Avg SSIM:     {avg_ssim:.4f}")
        click.echo(f"Avg Coverage: {avg_coverage:.1%}")
        click.echo(f"Avg Layers:   {avg_layers:.0f}")
        click.echo(f"Avg Time:     {avg_time:.1f}s")

    # Save reports
    report_data = {
        "total": len(results),
        "success": len(success_results),
        "total_time_s": round(total_elapsed, 1),
        "config": {
            "device": device,
            "segmentation_backend": segmentation_backend,
        },
        "results": results,
    }

    if report_format in ("json", "both"):
        json_path = output_path / "report.json"
        json_path.write_text(json.dumps(report_data, indent=2, ensure_ascii=False))
        click.echo(f"JSON report: {json_path}")

    if report_format in ("csv", "both"):
        csv_path = output_path / "report.csv"
        if success_results:
            fields = list(success_results[0].keys())
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=fields)
                writer.writeheader()
                for r in results:
                    writer.writerow({k: r.get(k, "") for k in fields})
            click.echo(f"CSV report:  {csv_path}")


if __name__ == "__main__":
    evaluate()
