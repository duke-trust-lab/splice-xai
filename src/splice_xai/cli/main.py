#!/usr/bin/env python
"""Command-line interface for SPLICE-XAI"""

import argparse
import logging
import sys
from pathlib import Path
import random

import numpy as np

from splice_xai import SPLICEAnalyzer, InpaintingConfig
from splice_xai.visualization.plotting import create_comparison_plot
from splice_xai.utils.file_io import save_results_to_csv

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _validate_paths(args) -> None:
    yolo = Path(args.yolo_model)
    if not yolo.is_file():
        raise FileNotFoundError(f"--yolo-model not found: {yolo}")

    if args.image:
        img = Path(args.image)
        if not img.is_file():
            raise FileNotFoundError(f"--image not found: {img}")
    else:
        batch_dir = Path(args.batch)
        if not batch_dir.is_dir():
            raise NotADirectoryError(f"--batch directory not found: {batch_dir}")

    if args.mask:
        mask = Path(args.mask)
        if not mask.is_file():
            raise FileNotFoundError(f"--mask not found: {mask}")

    if args.csv:
        csv_path = Path(args.csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)
    except Exception:
        pass


def _conf_tuple(result):
    """
    Normalize confidence logging. Adjust to your canonical schema once decided.
    Supports both (confidence_before/after) and (original_confidence/result_confidence).
    """
    before = getattr(result, "confidence_before", None)
    after = getattr(result, "confidence_after", None)
    if before is None and after is None:
        before = getattr(result, "original_confidence", None)
        after = getattr(result, "result_confidence", None)
    return before, after


def main():
    parser = argparse.ArgumentParser(
        description="SPLICE-XAI: Generate counterfactual explanations"
    )

    # Inputs
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to a single image")
    input_group.add_argument("--batch", help="Path to a directory of images")

    # Model
    parser.add_argument("--yolo-model", required=True, help="Path to YOLO model (.pt)")
    parser.add_argument("--mask", help="Path to mask image")

    # Experiment
    parser.add_argument(
        "--mode",
        choices=["remove", "replace", "background", "all"],
        default="remove",
        help="Counterfactual generation mode",
    )

    # Inpainting
    parser.add_argument(
        "--inpaint-model",
        choices=["lama", "stable_diffusion", "flux", "sdxl"],
        default="stable_diffusion",
        help="Inpainting model to use",
    )
    parser.add_argument("--prompt", help="Custom prompt for inpainting / replacement")
    parser.add_argument("--negative-prompt", help="Negative prompt")
    parser.add_argument("--background", default="rocky", help="Background type")
    parser.add_argument("--target-label", help="Target class for replacement")

    # Output
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument("--save-images", action="store_true", help="Save generated images")
    parser.add_argument("--visualize", action="store_true", help="Create visualizations")
    parser.add_argument("--csv", help="Path to save CSV results")

    # Config
    parser.add_argument("--conf-threshold", type=float, default=0.4, help="YOLO confidence threshold")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device for detection/inference",
    )

    args = parser.parse_args()

    # Early checks
    if args.mode in ("replace", "all") and not args.prompt:
        logger.warning("Mode includes 'replace' but --prompt is empty. Replacement runs will be skipped.")

    try:
        _validate_paths(args)
    except Exception as e:
        logger.error(e)
        return 2

    _set_seed(args.seed)

    # Initialize analyzer
    config = InpaintingConfig(yolo_conf_threshold=args.conf_threshold, device=args.device if hasattr(InpaintingConfig, "device") else None)
    try:
        analyzer = SPLICEAnalyzer(args.yolo_model, config)
    except Exception as e:
        logger.error(f"Failed to initialize analyzer: {e}")
        return 1

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    results = []

    def _run_single(image_path: str, mode: str):
        try:
            if mode == "remove":
                return analyzer.remove_object(
                    image_path,
                    mask_path=args.mask,
                    prompt=args.prompt,
                    model=args.inpaint_model,
                    negative_prompt=args.negative_prompt,
                )
            if mode == "replace":
                if not args.prompt:
                    logger.error("Replacement mode requires --prompt; skipping.")
                    return None
                return analyzer.replace_object(
                    image_path,
                    replacement_prompt=args.prompt,
                    mask_path=args.mask,
                    model=args.inpaint_model,
                    target_label=args.target_label,
                    negative_prompt=args.negative_prompt,
                )
            if mode == "background":
                return analyzer.change_background(
                    image_path,
                    background=args.background,
                    mask_path=args.mask,
                    model=args.inpaint_model,
                )
        except Exception as e:
            logger.error(f"Error processing {image_path} [{mode}]: {e}")
            return None
        return None

    # Single image
    if args.image:
        modes = ["remove", "replace", "background"] if args.mode == "all" else [args.mode]
        for mode in modes:
            logger.info(f"Processing {args.image} with mode: {mode}")
            result = _run_single(args.image, mode)
            if result is None:
                continue
            results.append(result)

            # Save outputs
            if args.save_images and getattr(result, "image", None) is not None:
                img_path = output_dir / f"{Path(args.image).stem}_{mode}.png"
                result.image.save(img_path)
                setattr(result, "output_image_path", str(img_path))
                logger.info(f"Saved image to {img_path}")
            elif args.save_images:
                logger.warning(f"No image generated for {args.image} [{mode}]")

            if args.visualize and getattr(result, "image", None) is not None:
                viz_path = output_dir / f"{Path(args.image).stem}_{mode}_viz.png"
                create_comparison_plot(args.image, result, str(viz_path))
                setattr(result, "visualization_path", str(viz_path))

            before, after = _conf_tuple(result)
            if before is not None and after is not None:
                logger.info(f"Result: {result.outcome} (confidence: {before:.2f} -> {after:.2f})")
            else:
                logger.info(f"Result: {result.outcome} (no confidence values available)")

    # Batch directory
    else:
        logger.info(f"Batch processing directory: {args.batch}")
        # Build mode list
        modes = (
            ["remove", "replace", "background"]
            if args.mode == "all"
            else [args.mode]
        )

        # Collect images
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        img_paths = [p for p in Path(args.batch).rglob("*") if p.suffix.lower() in exts]
        if not img_paths:
            logger.warning("No images found in batch directory.")
        for img in img_paths:
            for mode in modes:
                logger.info(f"Processing {img} with mode: {mode}")
                r = _run_single(str(img), mode)
                if r is None:
                    continue
                results.append(r)

                if args.save_images and getattr(r, "image", None) is not None:
                    img_path = output_dir / f"{img.stem}_{mode}.png"
                    r.image.save(img_path)
                    setattr(r, "output_image_path", str(img_path))
                    logger.info(f"Saved image to {img_path}")
                elif args.save_images:
                    logger.warning(f"No image generated for {img} [{mode}]")

                if args.visualize and getattr(r, "image", None) is not None:
                    viz_path = output_dir / f"{img.stem}_{mode}_viz.png"
                    create_comparison_plot(str(img), r, str(viz_path))
                    setattr(r, "visualization_path", str(viz_path))

                before, after = _conf_tuple(r)
                if before is not None and after is not None:
                    logger.info(f"Result: {r.outcome} (confidence: {before:.2f} -> {after:.2f})")
                else:
                    logger.info(f"Result: {r.outcome} (no confidence values available)")

    # CSV
    if args.csv and results:
        try:
            csv_data = [r.to_dict() for r in results]
            save_results_to_csv(csv_data, args.csv)
            logger.info(f"Saved results to {args.csv}")
        except Exception as e:
            logger.error(f"Failed to save CSV: {e}")

    successful = sum(1 for r in results if getattr(r, "success", False))
    logger.info(f"\nProcessing complete: {successful}/{len(results)} successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
