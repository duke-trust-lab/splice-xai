import time
import logging
from pathlib import Path
from typing import Optional, Dict, Any, List

import numpy as np
from PIL import Image

from .config import InpaintingConfig
from .results import CounterfactualResult
from ..detection.yolo_detector import YOLODetector
from ..detection.fasterrcnn_detector import FasterRCNNDetector
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
        model_path: str,
        config: Optional[InpaintingConfig] = None,
        replicate_api_key: Optional[str] = None,
        model_type: str = "yolo",
        num_classes: int = 1,
    ):
        """
        Initializes the SPLICE Analyzer with an adaptive detector backend.
        """
        self.config = config or InpaintingConfig()
        device = getattr(self.config, "device", "auto")

        # --- Adaptive Detector Factory ---
        if model_type.lower() == "yolo":
            logger.info(f"Initializing YOLO detector from {model_path}")
            self.detector = YOLODetector(model_path, device=device)
        elif model_type.lower() == "frcnn":
            logger.info(f"Initializing Faster R-CNN detector from {model_path}")
            self.detector = FasterRCNNDetector(
                model_path, device=device, num_classes=num_classes
            )
        else:
            raise ValueError(
                f"Unsupported model_type: {model_type}. Use 'yolo' or 'frcnn'."
            )

        # --- Conditional SAM Initialization ---
        self.segmentor = None
        if self.config.use_sam:
            self.segmentor = SAMSegmentor(
                checkpoint_path=self.config.sam_checkpoint_path,
                checkpoint_url=self.config.sam_checkpoint_url,
                device=device,
            )
        else:
            logger.info("SAM initialization skipped (using Bounding Box mask mode).")

        self.inpainter = ReplicateInpainter(replicate_api_key, self.config)

    # ---------------------------
    # Single image methods
    # ---------------------------
    def remove_object(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        prompt: Optional[str] = None,
        model: str = "stable_diffusion",
        mask_mode: Optional[str] = None,
        **kwargs,
    ) -> CounterfactualResult:
        """Generate a counterfactual by removing detected object(s)."""
        start_time = time.time()
        image = load_image(image_path)
        effective_mask_mode = mask_mode or getattr(self.config, "mask_mode", "top1")

        # 1. Mask Generation
        mask = self._generate_mask(image, mode=effective_mask_mode)
        if mask is None:
            return CounterfactualResult(
                experiment_type="remove",
                image_path=image_path,
                success=False,
                outcome="no mask",
            )

        # 2. Original Detection
        orig_detection = self.detector.get_top_detection(
            np.array(image), self.config.detector_conf_threshold
        )
        if not orig_detection.has_detection:
            return CounterfactualResult(
                experiment_type="remove",
                image_path=image_path,
                success=False,
                outcome="no detections",
            )

        # --- FIX: Robust Instance Data Collection ---
        all_boxes = orig_detection.full_results.get("all_boxes", [])
        original_confs = []
        original_cls_ids = []
        original_boxes = []

        for box in all_boxes:
            # Ensure we don't crash if box is shorter than expected
            original_boxes.append([float(x) for x in box[:4]])
            # Fallback to top detection values if per-box data is missing
            conf = float(box[4]) if len(box) > 4 else float(orig_detection.confidence)
            cls_id = int(box[5]) if len(box) > 5 else int(orig_detection.class_id)
            original_confs.append(conf)
            original_cls_ids.append(cls_id)

        # 3. Inpainting
        img_resized, mask_resized = resize_for_model(
            image, mask, model, self.config.default_model_sizes
        )
        with GPUResourceManager():
            inpainted = self.inpainter.inpaint(
                img_resized,
                mask_resized,
                model,
                prompt or self.config.default_removal_prompt,
                negative_prompt=kwargs.get("negative_prompt")
                or self.config.default_negative_prompt
                or "",
            )

        inpainted = inpainted.resize(image.size, Image.LANCZOS)

        # 4. Results
        result_detection = self.detector.get_top_detection(
            np.array(inpainted), self.config.detector_conf_threshold
        )
        result_count = len(result_detection.full_results.get("all_boxes", []))

        outcome = (
            "success"
            if result_count == 0
            else (
                "partial_success" if result_count < len(all_boxes) else "unsuccessful"
            )
        )

        return CounterfactualResult(
            image=inpainted,
            mask=mask,
            success=(outcome == "success"),
            outcome=outcome,
            mask_mode=effective_mask_mode,
            experiment_type="remove",
            image_path=image_path,
            original_count=len(all_boxes),
            original_confidence=orig_detection.confidence,
            original_confs=original_confs,
            original_cls_ids=original_cls_ids,
            original_boxes=original_boxes,
            result_count=result_count,
            result_confidence=result_detection.confidence,
            runtime_seconds=time.time() - start_time,
        )

    def _generate_mask(
        self, image: Image.Image, mode: str = "top1"
    ) -> Optional[Image.Image]:
        """Handles the choice between SAM and Bounding Box masking with index safety."""
        detection = self.detector.get_top_detection(
            np.array(image), self.config.detector_conf_threshold
        )
        if not detection.has_detection:
            return None

        if mode == "union":
            boxes = detection.full_results.get("all_boxes", [])
        else:
            boxes = [detection.bbox] if detection.bbox is not None else []

        if not boxes:
            return None

        if self.config.use_sam and self.segmentor is not None:
            return self.segmentor.generate_mask(np.array(image), np.array(boxes), mode)
        else:
            # --- FIX: Box-Only logic with index safety and clipping ---
            width, height = image.size
            mask_np = np.zeros((height, width), dtype=np.uint8)
            for box in boxes:
                try:
                    # Only take first 4 elements regardless of list length
                    coords = [int(float(c)) for c in box[:4]]
                    x1, y1, x2, y2 = coords
                    # Boundary clipping
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)
                    mask_np[y1:y2, x1:x2] = 255
                except (ValueError, IndexError):
                    continue
            return Image.fromarray(mask_np)

    def replace_object(
        self,
        image_path: str,
        replacement_prompt: str,
        mask_path: Optional[str] = None,
        model: str = "stable_diffusion",
        target_label: Optional[str] = None,
        mask_mode: Optional[str] = None,
        **kwargs,
    ) -> CounterfactualResult:
        """Generate a counterfactual by replacing detected object(s)."""
        result = self.remove_object(
            image_path,
            mask_path=mask_path,
            prompt=replacement_prompt,
            model=model,
            mask_mode=mask_mode,
            **kwargs,
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
        bg_prompt = self.config.available_backgrounds.get(
            background, "natural background"
        )

        mask = self._generate_mask(image, mode="union")
        if mask is None:
            return None

        alpha = self._extract_object_alpha(image, mask)
        bg_image = self._generate_background(bg_prompt, image.size, model, **kwargs)
        result_image = self._composite_over_background(image, alpha, bg_image)

        result_detection = self.detector.get_top_detection(
            np.array(result_image), self.config.detector_conf_threshold
        )

        return CounterfactualResult(
            image=result_image,
            mask=alpha,
            success=True,
            outcome="success",
            experiment_type="background",
            image_path=image_path,
            inpainting_model=model,
            positive_prompt=bg_prompt,
            background_type=background,
            original_count=len(
                self.detector.get_top_detection(
                    np.array(image), self.config.detector_conf_threshold
                ).full_results.get("all_boxes", [])
            ),
            result_count=len(result_detection.full_results.get("all_boxes", [])),
            runtime_seconds=time.time() - start_time,
        )

    # ---------------------------
    # Helpers
    # ---------------------------
    def _generate_mask(
        self, image: Image.Image, mode: str = "top1"
    ) -> Optional[Image.Image]:
        """Handles the choice between SAM and Bounding Box masking."""
        detection = self.detector.get_top_detection(
            np.array(image), self.config.detector_conf_threshold
        )
        if not detection.has_detection:
            return None

        # Gather relevant boxes based on mode
        if mode == "union":
            boxes = detection.full_results.get("all_boxes", [])
        else:
            boxes = [detection.bbox] if detection.bbox is not None else []

        if not boxes:
            return None

        if self.config.use_sam and self.segmentor is not None:
            logger.info(f"Generating mask via SAM ({mode})")
            return self.segmentor.generate_mask(np.array(image), np.array(boxes), mode)
        else:
            logger.info(f"Generating mask via Bounding Box ({mode})")
            width, height = image.size
            mask_np = np.zeros((height, width), dtype=np.uint8)

            for box in boxes:
                # Robust coordinate extraction: take only the first 4 elements
                # regardless of whether the list has 4, 5, or 6 items.
                try:
                    x1, y1, x2, y2 = map(int, box[:4])

                    # Clip coordinates to image boundaries
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(width, x2), min(height, y2)

                    # Fill the rectangle
                    mask_np[y1:y2, x1:x2] = 255
                except (IndexError, TypeError) as e:
                    logger.warning(f"Skipping malformed box {box}: {e}")
                    continue

            return Image.fromarray(mask_np)

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
        c_res, m_res = resize_for_model(
            canvas, full_mask, model, self.config.default_model_sizes
        )

        bg = self.inpainter.inpaint(
            c_res,
            m_res,
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
