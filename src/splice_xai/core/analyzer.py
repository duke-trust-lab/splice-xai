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
        self.config = config or InpaintingConfig()
        device = getattr(self.config, "device", "auto")

        if model_type.lower() == "yolo":
            logger.info(f"Initializing YOLO detector from {model_path}")
            self.detector = YOLODetector(model_path, device=device)
        elif model_type.lower() == "frcnn":
            logger.info(f"Initializing Faster R-CNN detector from {model_path}")
            self.detector = FasterRCNNDetector(
                model_path, device=device, num_classes=num_classes
            )
        else:
            raise ValueError(f"Unsupported model_type: {model_type}")

        self.segmentor = None
        if self.config.use_sam:
            self.segmentor = SAMSegmentor(
                checkpoint_path=self.config.sam_checkpoint_path,
                checkpoint_url=self.config.sam_checkpoint_url,
                device=device,
            )

        self.inpainter = ReplicateInpainter(replicate_api_key, self.config)

    def remove_object(
        self,
        image_path: str,
        mask_path: Optional[str] = None,
        prompt: Optional[str] = None,
        model: str = "stable_diffusion",
        mask_mode: Optional[str] = None,
        **kwargs,
    ) -> CounterfactualResult:
        """
        Generate a counterfactual with exclusive instance tracking and
        class-aware success verification.
        """
        start_time = time.time()

        # 1. Original Detection (Before)
        orig_detection = self.detector.get_top_detection(
            image_path, self.config.detector_conf_threshold
        )

        image = load_image(image_path)
        effective_mask_mode = mask_mode or getattr(self.config, "mask_mode", "top1")

        # 2. Mask Generation
        mask = self._generate_mask(image, orig_detection, mode=effective_mask_mode)

        if mask is None:
            return CounterfactualResult(
                experiment_type="remove",
                image_path=image_path,
                success=False,
                outcome="no mask",
            )

        if not orig_detection.has_detection:
            return CounterfactualResult(
                experiment_type="remove",
                image_path=image_path,
                success=False,
                outcome="no detections",
            )

        # Sync original instance data for per-instance tracking
        full_res = orig_detection.full_results
        original_boxes = full_res.get("all_boxes", [])
        original_confs = full_res.get("all_confs", [])
        original_cls_ids = full_res.get("all_cls_ids", [])

        # Track which classes we are specifically trying to remove/replace
        target_class_ids = set(original_cls_ids)

        # 3. Inpainting Pipeline
        img_resized, mask_resized = resize_for_model(
            image, mask, model, self.config.default_model_sizes
        )

        with GPUResourceManager():
            inpainted_raw = self.inpainter.inpaint(
                img_resized,
                mask_resized,
                model,
                prompt or self.config.default_removal_prompt,
                negative_prompt=kwargs.get("negative_prompt") or "",
            )

        # 4. Post-Processing: Restore original dimensions
        inpainted = inpainted_raw.resize(image.size, Image.LANCZOS)

        # Ensure array parity for the "After" detection
        result_img_array = np.ascontiguousarray(
            np.array(inpainted.convert("RGB")), dtype=np.uint8
        )

        # --- EXCLUSIVE INSTANCE-SPECIFIC MAPPING LOGIC ---
        # Get ALL raw detections (low threshold) to find "ghost" artifacts
        after_raw = self.detector.get_top_detection(
            result_img_array, conf_threshold=0.01
        )
        after_boxes = np.array(after_raw.full_results.get("all_boxes", []))
        after_confs = np.array(after_raw.full_results.get("all_confs", []))

        result_confs_mapped = []
        claimed_after_indices = set()  # Lock detections once assigned

        for org_box in original_boxes:
            org_cx = (org_box[0] + org_box[2]) / 2
            org_cy = (org_box[1] + org_box[3]) / 2

            best_match_conf = 0.0
            best_match_idx = -1
            min_dist = float("inf")

            # Proximity threshold based on the original object's dimensions
            prox_limit = max(org_box[2] - org_box[0], org_box[3] - org_box[1]) * 1.5

            if len(after_boxes) > 0:
                for i, aft_box in enumerate(after_boxes):
                    if i in claimed_after_indices:
                        continue

                    aft_cx = (aft_box[0] + aft_box[2]) / 2
                    aft_cy = (aft_box[1] + aft_box[3]) / 2

                    dist = np.sqrt((org_cx - aft_cx) ** 2 + (org_cy - aft_cy) ** 2)

                    # Update best match if it's the closest one within proximity
                    if dist < prox_limit and dist < min_dist:
                        min_dist = dist
                        best_match_conf = after_confs[i]
                        best_match_idx = i

            if best_match_idx != -1:
                claimed_after_indices.add(best_match_idx)

            result_confs_mapped.append(float(best_match_conf))

        # 5. Class-Aware Success Check
        # Run detection at standard threshold to check for lingering target objects
        success_detection = self.detector.get_top_detection(
            result_img_array, conf_threshold=self.config.detector_conf_threshold
        )

        after_full_res = success_detection.full_results
        after_cls_ids = after_full_res.get("all_cls_ids", [])

        # Count only if the remaining detection matches one of the original classes
        # This ignores newly correctly identified replacement objects (e.g., 'rock')
        remaining_targets = [cid for cid in after_cls_ids if cid in target_class_ids]

        result_count = len(remaining_targets)
        is_success = result_count == 0

        # Define scientific outcome
        if is_success:
            outcome = "success"
        else:
            outcome = (
                "partial_success"
                if result_count < len(original_boxes)
                else "unsuccessful"
            )

        return CounterfactualResult(
            image=inpainted,
            mask=mask,
            success=is_success,
            outcome=outcome,
            use_sam=self.config.use_sam,
            mask_mode=effective_mask_mode,
            experiment_type="remove",
            image_path=image_path,
            inpainting_model=model,
            original_count=len(original_boxes),
            original_confidence=orig_detection.confidence,
            original_confs=original_confs,
            original_cls_ids=original_cls_ids,
            original_boxes=original_boxes,
            # Tracks only the target class population change
            result_count=result_count,
            result_confs=result_confs_mapped,
            result_confidence=(
                float(np.max(result_confs_mapped)) if result_confs_mapped else 0.0
            ),
            runtime_seconds=time.time() - start_time,
        )

    def _generate_mask(
        self, image: Image.Image, detection: Any, mode: str = "top1"
    ) -> Optional[Image.Image]:
        if detection is None or not detection.has_detection:
            return None

        boxes = (
            detection.full_results.get("all_boxes", [])
            if mode == "union"
            else ([detection.bbox] if detection.bbox else [])
        )
        if not boxes:
            return None

        width, height = image.size

        if self.config.use_sam and self.segmentor is not None:
            img_np = np.ascontiguousarray(
                np.array(image.convert("RGB")), dtype=np.uint8
            )
            return self.segmentor.generate_mask(
                img_np, np.array(boxes, dtype=np.float32), mode
            )
        else:
            mask_np = np.zeros((height, width), dtype=np.uint8)
            for box in boxes:
                try:
                    x1, y1, x2, y2 = map(float, [float(c) for c in box[:4]])
                    w, h = x2 - x1, y2 - y1

                    # 15% Area Expansion Logic:
                    # To increase area by 15%, sides increase by sqrt(1.15) ~ 1.072
                    # We shift each edge out by 3.6% of the dimension
                    ix1 = max(0, int(x1 - (w * 0.036)))
                    iy1 = max(0, int(y1 - (h * 0.036)))
                    ix2 = min(width, int(x2 + (w * 0.036)))
                    iy2 = min(height, int(y2 + (h * 0.036)))

                    mask_np[iy1:iy2, ix1:ix2] = 255
                except:
                    continue
            return Image.fromarray(mask_np, mode="L")

    def replace_object(
        self, image_path: str, replacement_prompt: str, **kwargs
    ) -> CounterfactualResult:
        is_multi = kwargs.get("remove_all", False) or kwargs.get("replace_all", False)

        if is_multi:
            kwargs["mask_mode"] = "union"

        # CRITICAL: Pass replacement_prompt as the 'prompt' argument
        result = self.remove_object(image_path, prompt=replacement_prompt, **kwargs)

        result.experiment_type = "replace"
        result.positive_prompt = replacement_prompt

        # Logic for outcome strings...
        if result.success:
            result.outcome = "all_replaced" if is_multi else "replaced"
        return result

    def change_background(
        self, image_path: str, background: str, **kwargs
    ) -> CounterfactualResult:
        """
        Swaps background while tracking original vs. hallucinated detections.
        Applies 20% area expansion if box-only flag is set.
        """
        start_time = time.time()
        image = load_image(image_path)

        # 1. PRE-PROCESS: Original Detection
        orig_det = self.detector.get_top_detection(
            image_path, self.config.detector_conf_threshold
        )
        if not orig_det.has_detection:
            return CounterfactualResult(
                experiment_type="background",
                image_path=image_path,
                success=False,
                outcome="no detections",
            )

        original_boxes = orig_det.full_results.get("all_boxes", [])
        original_confs = orig_det.full_results.get("all_confs", [])
        original_cls_ids = orig_det.full_results.get("all_cls_ids", [])

        # 2. CORE LOGIC: Mask & Expansion
        # _generate_mask automatically applies the 20% area expansion (4.75% shift) if SAM is off
        mask = self._generate_mask(image, orig_det, mode="union")
        if mask is None:
            return None

        alpha = self._extract_object_alpha(image, mask)
        bg_prompt = self.config.available_backgrounds.get(
            background, "natural background"
        )
        bg_image = self._generate_background(
            bg_prompt, image.size, kwargs.get("model", "stable_diffusion")
        )

        # Composite original seals (with 20% buffer) over the new background
        result_image = self._composite_over_background(image, alpha, bg_image)

        # 3. POST-PROCESS: Evaluation
        result_img_array = np.ascontiguousarray(
            np.array(result_image.convert("RGB")), dtype=np.uint8
        )

        # Final detections at standard threshold
        success_detection = self.detector.get_top_detection(
            result_img_array, self.config.detector_conf_threshold
        )
        after_boxes = np.array(success_detection.full_results.get("all_boxes", []))
        after_confs = np.array(success_detection.full_results.get("all_confs", []))
        after_cls_ids = np.array(success_detection.full_results.get("all_cls_ids", []))

        # 4. MAPPING: Match "After" to "Original" and find Hallucinations
        result_confs_mapped = []
        claimed_after_indices = set()

        # Phase A: Map Original Instances
        for org_box in original_boxes:
            org_cx, org_cy = (org_box[0] + org_box[2]) / 2, (
                org_box[1] + org_box[3]
            ) / 2
            best_match_conf, best_match_idx, min_dist = 0.0, -1, float("inf")
            prox_limit = max(org_box[2] - org_box[0], org_box[3] - org_box[1]) * 1.5

            for i, aft_box in enumerate(after_boxes):
                if i in claimed_after_indices:
                    continue
                aft_cx, aft_cy = (aft_box[0] + aft_box[2]) / 2, (
                    aft_box[1] + aft_box[3]
                ) / 2
                dist = np.sqrt((org_cx - aft_cx) ** 2 + (org_cy - aft_cy) ** 2)

                if dist < prox_limit and dist < min_dist:
                    min_dist, best_match_conf, best_match_idx = dist, after_confs[i], i

            if best_match_idx != -1:
                claimed_after_indices.add(best_match_idx)
            result_confs_mapped.append(float(best_match_conf))

        # Phase B: Identify Hallucinations (Unclaimed detections)
        additional_detections = []
        for i in range(len(after_boxes)):
            if i not in claimed_after_indices:
                additional_detections.append(
                    {"conf": float(after_confs[i]), "cls_id": int(after_cls_ids[i])}
                )

        # 5. CONSTRUCT RESULT
        return CounterfactualResult(
            image=result_image,
            mask=alpha,
            success=True,
            outcome="success",
            use_sam=self.config.use_sam,  # Required for plotting dashed boxes
            experiment_type="background",
            image_path=image_path,
            original_count=len(original_boxes),
            original_confs=original_confs,
            original_boxes=original_boxes,  # Required for Panel 1 plotting
            original_cls_ids=original_cls_ids,
            result_count=len(after_boxes),
            result_boxes=after_boxes.tolist(),  # Required for Panel 3 plotting
            result_confs=result_confs_mapped,
            additional_detections=additional_detections,  # Required for to_rows CSV
            runtime_seconds=time.time() - start_time,
        )

    def _extract_object_alpha(
        self, image: Image.Image, mask: Image.Image
    ) -> Image.Image:
        mask = dilate_mask(mask, pixels=3)
        mask = feather_mask(mask, radius=2)
        return mask

    def _generate_background(self, prompt: str, size: tuple, model: str) -> Image.Image:
        canvas = Image.new("RGB", size, (128, 128, 128))
        full_mask = Image.new("L", size, 255)
        c_res, m_res = resize_for_model(
            canvas, full_mask, model, self.config.default_model_sizes
        )
        bg = self.inpainter.inpaint(c_res, m_res, model, prompt)
        return bg.resize(size, Image.LANCZOS)

    def _composite_over_background(
        self, image: Image.Image, alpha: Image.Image, background: Image.Image
    ) -> Image.Image:
        fg = image.copy().convert("RGBA")
        fg.putalpha(alpha)
        bg = background.convert("RGBA")
        return Image.alpha_composite(bg, fg).convert("RGB")
