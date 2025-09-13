import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

from .config import InpaintingConfig
from .results import CounterfactualResult
from ..detection.yolo_detector import YOLODetector
from ..detection.sam_segmentation import SAMSegmentor
from ..inpainting.replicate_backend import ReplicateInpainter
from ..utils.image_processing import dilate_mask, feather_mask, resize_for_model
from ..utils.file_io import load_image, load_mask
from ..utils.gpu_manager import GPUResourceManager

logger = logging.getLogger(__name__)


class SPLICEAnalyzer:
    """Main SPLICE analyzer for generating counterfactual explanations."""

    def __init__(
        self,
        yolo_model: str,
        config: Optional[InpaintingConfig] = None,
        replicate_api_key: Optional[str] = None,
    ):
        self.config = config or InpaintingConfig()
        self.detector = YOLODetector(yolo_model, device=getattr(self.config, "device", "auto"))
        self.segmentor = SAMSegmentor(
            checkpoint_path=self.config.sam_checkpoint_path,
            checkpoint_url=self.config.sam_checkpoint_url,
            device=getattr(self.config, "device", "auto"),
        )
        self.inpainter = ReplicateInpainter(replicate_api_key, self.config)

        logger.info(
            f"SPLICE Analyzer initialized with {len(self.detector.class_names)} classes"
        )

    # ---------------------------
    # Single image methods
    # ---------------------------
    def remove_object(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        prompt: Optional[str] = None,
        model: str = "stable_diffusion",
        **kwargs,
    ) -> CounterfactualResult:
        """Generate a counterfactual by removing the top detected object."""
        start_time = time.time()

        image = load_image(image_path)

        # Mask
        if mask_path:
            mask = load_mask(mask_path)
        else:
            mask = self._generate_mask(image)
            if mask is None:
                return CounterfactualResult(
                    experiment_type="remove",
                    image_path=image_path,
                    success=False,
                    outcome="unsuccessful",
                )

        # Detection (original)
        orig_detection = self.detector.get_top_detection(
            np.array(image), self.config.yolo_conf_threshold
        )
        if not orig_detection.has_detection:
            return CounterfactualResult(
                experiment_type="remove",
                image_path=image_path,
                success=False,
                outcome="unsuccessful",
            )

        # Prompt
        if not prompt:
            prompt = self.config.default_removal_prompt

        # Resize + inpaint
        img_resized, mask_resized = resize_for_model(
            image, mask, model, self.config.default_model_sizes
        )

        with GPUResourceManager():
            neg_prompt = kwargs.pop("negative_prompt", None)
            if neg_prompt is None:
                neg_prompt = self.config.default_negative_prompt or ""

            inpainted = self.inpainter.inpaint(
                img_resized,
                mask_resized,
                model,
                prompt,
                negative_prompt=neg_prompt,
                **kwargs,
            )

        inpainted = inpainted.resize(image.size, Image.LANCZOS)

        # Detection (result)
        result_detection = self.detector.get_top_detection(
            np.array(inpainted), self.config.yolo_conf_threshold
        )

        orig_boxes = orig_detection.full_results.get("all_boxes", []) or []
        result_boxes = result_detection.full_results.get("all_boxes", []) or []

        orig_count = len(orig_boxes)
        result_count = len(result_boxes)

        if not result_detection.has_detection:
            outcome = "success"
        elif result_count < orig_count:
            outcome = "partial_success"
        else:
            outcome = "unsuccessful"

        return CounterfactualResult(
            image=inpainted,
            mask=mask,
            success=(outcome == "success"),
            outcome=outcome,
            experiment_type="remove",
            image_path=image_path,
            mask_path=mask_path,
            inpainting_model=model,
            positive_prompt=prompt,
            negative_prompt=neg_prompt,  # return the effective value used
            original_class_id=orig_detection.class_id,
            original_class_name=self.detector.class_names.get(orig_detection.class_id),
            original_confidence=orig_detection.confidence,
            original_bbox=orig_detection.bbox,
            original_count=orig_count,
            original_boxes=orig_boxes,
            result_class_id=result_detection.class_id,
            result_class_name=self.detector.class_names.get(result_detection.class_id),
            result_confidence=result_detection.confidence,
            result_bbox=result_detection.bbox,
            result_count=result_count,
            result_boxes=result_boxes,
            runtime_seconds=time.time() - start_time,
        )

    def replace_object(
        self,
        image_path: str,
        replacement_prompt: str,
        mask_path: Optional[str] = None,
        model: str = "stable_diffusion",
        target_label: Optional[str] = None,
        **kwargs,
    ) -> CounterfactualResult:
        """Generate a counterfactual by replacing the top detected object."""
        result = self.remove_object(
            image_path, mask_path, replacement_prompt, model, **kwargs
        )
        result.experiment_type = "replace"
        result.target_label = target_label

        if target_label and result.result_class_name:
            if target_label.lower() in result.result_class_name.lower():
                result.outcome = "success"
                result.success = True

        return result

    def change_background(
        self,
        image_path: str,
        background: str,
        mask_path: Optional[str] = None,
        model: str = "stable_diffusion",
        **kwargs,
    ) -> CounterfactualResult:
        """Generate a counterfactual by changing the background behind the object."""
        start_time = time.time()
        
        image = load_image(image_path)

        if background not in self.config.available_backgrounds:
            logger.warning(f"Unknown background '{background}', using 'rocky'.")
            background = "rocky"
        bg_prompt = self.config.available_backgrounds[background]

        # Object matte (alpha)
        if mask_path:
            mask = load_mask(mask_path)
        else:
            mask = self._generate_mask(image, mode="union")
            if mask is None:
                return CounterfactualResult(
                    experiment_type="background",
                    image_path=image_path,
                    success=False,
                    outcome="unsuccessful",
                )

        orig_detection = self.detector.get_top_detection(
            np.array(image), self.config.yolo_conf_threshold
        )

        alpha = self._extract_object_alpha(image, mask)
        bg_image = self._generate_background(bg_prompt, image.size, model, **kwargs)
        result_image = self._composite_over_background(image, alpha, bg_image)

        result_detection = self.detector.get_top_detection(
            np.array(result_image), self.config.yolo_conf_threshold
        )

        orig_boxes = orig_detection.full_results.get("all_boxes", []) or []
        result_boxes = result_detection.full_results.get("all_boxes", []) or []

        orig_count, result_count = len(orig_boxes), len(result_boxes)

        if orig_count > 0 and result_count >= orig_count:
            outcome = "success"
        elif result_count > 0:
            outcome = "partial_success"
        else:
            outcome = "unsuccessful"

        return CounterfactualResult(
            image=result_image,
            mask=alpha,
            success=(outcome == "success"),
            outcome=outcome,
            experiment_type="background",
            image_path=image_path,
            mask_path=mask_path,
            inpainting_model=model,
            positive_prompt=bg_prompt,
            background_type=background,
            original_class_id=orig_detection.class_id,
            original_class_name=self.detector.class_names.get(orig_detection.class_id),
            original_confidence=orig_detection.confidence,
            original_count=orig_count,
            result_class_id=result_detection.class_id,
            result_class_name=self.detector.class_names.get(result_detection.class_id),
            result_confidence=result_detection.confidence,
            result_count=result_count,
            result_boxes=result_boxes,
            runtime_seconds=time.time() - start_time,
        )

    # ---------------------------
    # Batch processing
    # ---------------------------
    def batch_process(
        self,
        image_dir: str,
        experiment_configs: List[Dict[str, Any]],
        output_dir: Optional[str] = None,
    ) -> List[CounterfactualResult]:
        """Run multiple experiments over a directory of images.

        experiment_configs: list of dicts with key 'type' in {'remove','replace','background'}.
        """
        results: List[CounterfactualResult] = []

        # Recursive discovery across common image extensions
        exts = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
        image_paths = [p for p in Path(image_dir).rglob("*") if p.suffix.lower() in exts]

        for img_path in image_paths:
            for cfg in experiment_configs:
                exp_type = cfg.get("type", "remove")
                try:
                    if exp_type == "remove":
                        result = self.remove_object(
                            str(img_path),
                            mask_path=cfg.get("mask_path"),
                            prompt=cfg.get("prompt"),
                            model=cfg.get("model", "stable_diffusion"),
                            negative_prompt=cfg.get("negative_prompt"),
                        )
                    elif exp_type == "replace":
                        replacement_prompt = cfg.get("replacement_prompt")
                        if not replacement_prompt:
                            raise ValueError("replacement requires 'replacement_prompt'")
                        result = self.replace_object(
                            str(img_path),
                            replacement_prompt=replacement_prompt,
                            mask_path=cfg.get("mask_path"),
                            model=cfg.get("model", "stable_diffusion"),
                            target_label=cfg.get("target_label"),
                            negative_prompt=cfg.get("negative_prompt"),
                        )
                    elif exp_type == "background":
                        result = self.change_background(
                            str(img_path),
                            background=cfg.get("background", "rocky"),
                            mask_path=cfg.get("mask_path"),
                            model=cfg.get("model", "stable_diffusion"),
                        )
                    else:
                        logger.warning(f"Unknown experiment type '{exp_type}', skipping.")
                        continue

                    results.append(result)
                except Exception as e:
                    logger.error(f"Batch error on {img_path} [{cfg}]: {e}")
                    continue

        return results

    # ---------------------------
    # Helpers
    # ---------------------------
    def _generate_mask(
        self, image: Image.Image, mode: str = "top1"
    ) -> Optional[Image.Image]:
        detection = self.detector.get_top_detection(
            np.array(image), self.config.yolo_conf_threshold
        )
        if not detection.has_detection:
            return None

        if mode == "union":
            boxes = np.array(detection.full_results.get("all_boxes", []))
            if boxes.size == 0:
                # fallback to top box if available
                boxes = np.array([detection.bbox]) if detection.bbox is not None else np.array([])
        else:  # top1
            boxes = np.array([detection.bbox]) if detection.bbox is not None else np.array([])

        if boxes.size == 0:
            return None

        return self.segmentor.generate_mask(np.array(image), boxes, mode)

    def _extract_object_alpha(
        self, image: Image.Image, mask: Image.Image
    ) -> Image.Image:
        mask = dilate_mask(mask, pixels=3)
        mask = feather_mask(mask, radius=2)
        return mask

    def _generate_background(
        self, prompt: str, size: tuple, model: str = "stable_diffusion", **kwargs
    ) -> Image.Image:
        canvas = Image.new("RGB", size, (128, 128, 128))
        full_mask = Image.new("L", size, 255)

        canvas_resized, mask_resized = resize_for_model(
            canvas, full_mask, model, self.config.default_model_sizes
        )

        bg = self.inpainter.inpaint(
            canvas_resized,
            mask_resized,
            model,
            prompt,
            negative_prompt="people, animals, objects, text",
            **kwargs,
        )
        return bg.resize(size, Image.LANCZOS)

    def _composite_over_background(
        self, image: Image.Image, alpha: Image.Image, background: Image.Image
    ) -> Image.Image:
        fg = image.copy().convert("RGBA")
        fg.putalpha(alpha)
        bg = background.convert("RGBA")
        return Image.alpha_composite(bg, fg).convert("RGB")
