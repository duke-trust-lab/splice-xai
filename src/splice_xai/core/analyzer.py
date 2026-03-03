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

        Args:
            model_path: Path to the model weights (.pt or .pth).
            config: Configuration object.
            replicate_api_key: Optional API key for Replicate inpainting.
            model_type: Architecture type ('yolo' or 'frcnn').
            num_classes: Number of foreground classes (used for Faster R-CNN).
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

        logger.info(
            f"SPLICE Analyzer initialized with {len(self.detector.class_names)} classes "
            f"using {model_type.upper()} backend (Mask Mode: {'SAM' if self.config.use_sam else 'Bounding Box'})."
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

        # 1. Mask Generation
        if mask_path:
            mask = load_mask(mask_path)
        else:
            mask = self._generate_mask(image)
            if mask is None:
                return CounterfactualResult(
                    experiment_type="remove",
                    image_path=image_path,
                    success=False,
                    outcome="unsuccessful (no mask)",
                )

        # 2. Original Detection (Back-end Agnostic)
        orig_detection = self.detector.get_top_detection(
            np.array(image), self.config.detector_conf_threshold
        )
        if not orig_detection.has_detection:
            return CounterfactualResult(
                experiment_type="remove",
                image_path=image_path,
                success=False,
                outcome="unsuccessful (no original detection)",
            )

        # 3. Inpainting
        if not prompt:
            prompt = self.config.default_removal_prompt

        img_resized, mask_resized = resize_for_model(
            image, mask, model, self.config.default_model_sizes
        )

        with GPUResourceManager():
            neg_prompt = (
                kwargs.pop("negative_prompt", None)
                or self.config.default_negative_prompt
                or ""
            )
            inpainted = self.inpainter.inpaint(
                img_resized,
                mask_resized,
                model,
                prompt,
                negative_prompt=neg_prompt,
                **kwargs,
            )

        inpainted = inpainted.resize(image.size, Image.LANCZOS)

        # 4. Post-Inpainting Detection
        result_detection = self.detector.get_top_detection(
            np.array(inpainted), self.config.detector_conf_threshold
        )

        orig_boxes = orig_detection.full_results.get("all_boxes", []) or []
        result_boxes = result_detection.full_results.get("all_boxes", []) or []
        orig_count, result_count = len(orig_boxes), len(result_boxes)

        # Determine Success
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
            negative_prompt=neg_prompt,
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

        # Check if replacement caused the model to predict the target class
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
            logger.warning(f"Unknown background '{background}', using default.")
            background = next(iter(self.config.available_backgrounds))
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
                    outcome="unsuccessful (no mask)",
                )

        orig_detection = self.detector.get_top_detection(
            np.array(image), self.config.detector_conf_threshold
        )

        alpha = self._extract_object_alpha(image, mask)
        bg_image = self._generate_background(bg_prompt, image.size, model, **kwargs)
        result_image = self._composite_over_background(image, alpha, bg_image)

        result_detection = self.detector.get_top_detection(
            np.array(result_image), self.config.detector_conf_threshold
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
            boxes = np.array(detection.full_results.get("all_boxes", []))
            if boxes.size == 0 and detection.bbox:
                boxes = np.array([detection.bbox])
        else:  # top1
            boxes = np.array([detection.bbox]) if detection.bbox else np.array([])

        if boxes.size == 0:
            return None

        # --- BRANCH: SAM vs BOX ONLY ---
        if self.config.use_sam and self.segmentor is not None:
            logger.info("Generating mask via SAM (pixel-perfect)")
            return self.segmentor.generate_mask(np.array(image), boxes, mode)
        else:
            logger.info("Generating mask via Bounding Box (rectangular)")
            # Create a blank black mask (mode 'L' for 8-bit grayscale)
            width, height = image.size
            mask_np = np.zeros((height, width), dtype=np.uint8)

            # Fill the detected rectangles with white (255)
            for box in boxes:
                # box is [x1, y1, x2, y2]
                x1, y1, x2, y2 = map(int, box)

                # Clip coordinates to image boundaries to prevent indexing errors
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(width, x2), min(height, y2)

                # Fill the rectangle in the numpy array
                mask_np[y1:y2, x1:x2] = 255

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
