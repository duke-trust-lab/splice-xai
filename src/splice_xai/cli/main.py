#!/usr/bin/env python
"""Command-line interface for SPLICE-XAI supporting multiple detector backends"""

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

    # Adaptive Model Selection
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
    parser.add_argument(
        "--save-images", action="store_true", help="Save generated images"
    )
    parser.add_argument(
        "--visualize", action="store_true", help="Create visualizations"
    )
    parser.add_argument("--csv", help="Path to save CSV results")

    # Config
    parser.add_argument(
        "--conf-threshold",
        type=float,
        default=0.4,
        help="Detection confidence threshold",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Computation device",
    )

    args = parser.parse_args()

    # Determine Model Type Automatically
    m_type = args.model_type
    if m_type == "auto":
        m_type = "yolo" if args.model.lower().endswith(".pt") else "frcnn"

    logger.info(f"Using detector type: {m_type}")

    if args.mode in ("replace", "all") and not args.prompt:
        logger.warning(
            "Mode includes 'replace' but --prompt is empty. Replacement runs will be skipped."
        )

    try:
        _validate_paths(args)
    except Exception as e:
        logger.error(e)
        return 2

    _set_seed(args.seed)

    # Initialize analyzer with adaptive parameters
    config = InpaintingConfig(
        detector_conf_threshold=args.conf_threshold,
        device=args.device if hasattr(InpaintingConfig, "device") else None,
    )

    try:
        # We pass the model path, detected type, and num_classes to the SPLICEAnalyzer
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

    # Processing Logic
    input_paths = (
        [args.image]
        if args.image
        else [
            p
            for p in Path(args.batch).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        ]
    )

    modes = ["remove", "replace", "background"] if args.mode == "all" else [args.mode]

    for img_path in input_paths:
        for mode in modes:
            logger.info(f"Processing {img_path} with mode: {mode}")
            result = _run_single(str(img_path), mode)
            if result is None:
                continue
            results.append(result)

            if args.save_images and getattr(result, "image", None) is not None:
                stem = Path(img_path).stem
                out_name = f"{stem}_{mode}.png"
                result.image.save(output_dir / out_name)
                setattr(result, "output_image_path", str(output_dir / out_name))

            if args.visualize and getattr(result, "image", None) is not None:
                viz_path = output_dir / f"{Path(img_path).stem}_{mode}_viz.png"
                create_comparison_plot(str(img_path), result, str(viz_path))

            before, after = _conf_tuple(result)
            if before is not None and after is not None:
                logger.info(
                    f"Result: {result.outcome} (conf: {before:.2f} -> {after:.2f})"
                )

    if args.csv and results:
        save_results_to_csv([r.to_dict() for r in results], args.csv)

    successful = sum(1 for r in results if getattr(r, "success", False))
    logger.info(f"\nProcessing complete: {successful}/{len(results)} successful")
    return 0


if __name__ == "__main__":
    sys.exit(main())
