#!/usr/bin/env python
"""Command-line interface for SPLICE-XAI: Focused Replace & Flexible Remove"""

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
    model_path = Path(args.model)
    if not model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    if args.image:
        img = Path(args.image)
        if not img.is_file():
            raise FileNotFoundError(f"--image not found: {img}")
    else:
        batch_dir = Path(args.batch)
        if not batch_dir.is_dir():
            raise NotADirectoryError(f"--batch directory not found: {batch_dir}")

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
    before = getattr(result, "original_confidence", None)
    after = getattr(result, "result_confidence", None)
    return before, after


def main():
    parser = argparse.ArgumentParser(
        description="SPLICE-XAI: Generate counterfactual explanations"
    )

    # --- Inputs ---
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--image", help="Path to a single image")
    input_group.add_argument("--batch", help="Path to a directory of images")

    # --- Detection Model ---
    parser.add_argument(
        "--model", required=True, help="Path to model (.pt for YOLO, .pth for FRCNN)"
    )
    parser.add_argument(
        "--model-type",
        choices=["auto", "yolo", "frcnn"],
        default="auto",
        help="Model architecture type (default: auto-detect by extension)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=1,
        help="Number of foreground classes (required for Faster R-CNN)",
    )

    # --- Masking Options ---
    parser.add_argument(
        "--box-only",
        action="store_true",
        help="Use the detector bounding box as the mask instead of SAM pixel-mask",
    )
    parser.add_argument(
        "--remove-all",
        action="store_true",
        help="Target all detected objects (used for 'remove' mode)",
    )

    parser.add_argument(
        "--mask", help="Path to manual mask image (overrides auto-masking)"
    )

    # --- Experiment Mode ---
    parser.add_argument(
        "--mode",
        choices=["remove", "replace", "background", "all"],
        default="remove",
        help="Counterfactual generation mode",
    )

    # --- Inpainting ---
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

    # --- Output & Logging ---
    parser.add_argument("--output", default="results", help="Output directory")
    parser.add_argument(
        "--save-images", action="store_true", help="Save generated images"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualizations"
    )
    parser.add_argument("--csv", help="Path to save CSV results")

    # --- Config ---
    parser.add_argument(
        "--conf-threshold", type=float, default=0.4, help="Confidence threshold"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device",
    )

    parser.add_argument(
        "--perturbation",
        choices=[
            "inpainting",
            "black_mask",
            "blurring",
        ],
        default="inpainting",
        help=(
            "Removal method: diffusion inpainting, "
            "black animal mask, or masked Gaussian blur."
        ),
    )

    parser.add_argument(
        "--blur-radius",
        type=float,
        default=15.0,
        help="Gaussian blur radius in original-image pixels.",
    )

    parser.add_argument(
        "--save-masks",
        action="store_true",
        help="Save the actual mask used for each experiment.",
    )

    args = parser.parse_args()

    if args.perturbation != "inpainting" and args.mode != "remove":
        parser.error("black_mask and blurring require --mode remove.")

    if args.perturbation == "blurring":
        if not np.isfinite(args.blur_radius) or args.blur_radius <= 0:
            parser.error("--blur-radius must be finite and greater than zero.")

    # Determine Model Type Automatically
    m_type = args.model_type
    if m_type == "auto":
        m_type = "yolo" if args.model.lower().endswith(".pt") else "frcnn"

    logger.info(f"Using detector type: {m_type}")

    try:
        _validate_paths(args)
    except Exception as e:
        logger.error(e)
        return 2

    _set_seed(args.seed)

    # Initialize configuration: mask_mode is union only if explicitly removing all
    config = InpaintingConfig(
        detector_conf_threshold=args.conf_threshold,
        use_sam=not args.box_only,
        mask_mode="union" if (args.mode == "remove" and args.remove_all) else "top1",
        device=args.device,
    )

    try:
        analyzer = SPLICEAnalyzer(
            model_path=args.model,
            config=config,
            model_type=m_type,
            num_classes=args.num_classes,
        )
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
                    remove_all=args.remove_all,
                    perturbation=args.perturbation,
                    blur_radius=args.blur_radius,
                )
            if mode == "replace":
                return analyzer.replace_object(
                    image_path,
                    replacement_prompt=args.prompt,
                    mask_path=args.mask,
                    model=args.inpaint_model,
                    target_label=args.target_label,
                    negative_prompt=args.negative_prompt,
                    replace_all=False,  # STRICT: Always replace only the top detection
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

    input_paths = (
        [Path(args.image)]
        if args.image
        else [
            p
            for p in Path(args.batch).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
    )

    modes = ["remove", "replace", "background"] if args.mode == "all" else [args.mode]

    # Execution Loop
    for img_path in input_paths:
        for mode in modes:
            logger.info(f"Processing {img_path.name} | Mode: {mode}")
            result = _run_single(str(img_path), mode)
            if result is None:
                continue
            results.append(result)

            # Traditional outputs identify the perturbation instead of "remove".
            output_label = (
                args.perturbation
                if mode == "remove" and args.perturbation != "inpainting"
                else mode
            )

            stem = Path(img_path).stem

            if args.save_images and getattr(result, "image", None) is not None:
                image_path = output_dir / f"{stem}_{output_label}.png"

                result.image.save(image_path)
                result.output_image_path = str(image_path)

            if args.save_masks and getattr(result, "mask", None) is not None:
                mask_path = output_dir / f"{stem}_{output_label}_mask.png"

                result.mask.save(mask_path)

            if args.visualize and getattr(result, "image", None) is not None:
                viz_path = output_dir / f"{stem}_{output_label}_viz.png"

                create_comparison_plot(
                    str(img_path),
                    result,
                    str(viz_path),
                )

                result.visualization_path = str(viz_path)

            before, after = _conf_tuple(result)
            logger.info(
                f"Result: {result.outcome} ({result.original_count} -> {result.result_count})"
            )

    # --- CSV Generation ---
    # --- CSV Generation ---
    if args.csv and results:
        all_csv_rows = []

        for result in results:
            all_csv_rows.extend(result.to_rows())

        csv_path = Path(args.csv)

        if args.perturbation != "inpainting":
            suffix = f"_{args.perturbation}"

            if not csv_path.stem.endswith(suffix):
                csv_path = csv_path.with_name(
                    csv_path.stem + suffix + (csv_path.suffix or ".csv")
                )

        csv_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        save_results_to_csv(
            all_csv_rows,
            str(csv_path),
        )

        logger.info(
            "CSV saved to %s",
            csv_path,
        )

    successful = sum(1 for r in results if getattr(r, "success", False))
    logger.info(f"\nProcessing complete: {successful}/{len(results)} successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
