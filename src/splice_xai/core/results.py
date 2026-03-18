from dataclasses import dataclass, field
from typing import Optional, Any, Dict, List
from PIL import Image


@dataclass
class ObjectDetectionResult:
    bbox: Optional[List[float]] = None  # [x1, y1, x2, y2]
    class_id: Optional[int] = None
    confidence: Optional[float] = None
    full_results: Dict[str, Any] = field(default_factory=dict)

    @property
    def has_detection(self) -> bool:
        if self.bbox is not None:
            return True
        boxes = self.full_results.get("all_boxes")
        return bool(boxes)


@dataclass
class CounterfactualResult:
    image: Optional[Image.Image] = None
    mask: Optional[Image.Image] = None
    success: bool = False
    outcome: str = "unsuccessful"
    mask_mode: str = "top1"

    experiment_type: str = ""
    image_path: str = ""
    mask_path: Optional[str] = None
    inpainting_model: str = ""

    positive_prompt: Optional[str] = None
    negative_prompt: Optional[str] = None
    background_type: Optional[str] = None
    target_label: Optional[str] = None

    original_class_id: Optional[int] = None
    original_class_name: Optional[str] = None
    original_confidence: Optional[float] = None
    original_bbox: Optional[List[float]] = None
    original_count: int = 0

    result_class_id: Optional[int] = None
    result_class_name: Optional[str] = None
    result_confidence: Optional[float] = None
    result_bbox: Optional[List[float]] = None
    result_count: int = 0

    # Lists for granular per-instance tracking
    original_boxes: Optional[List[List[float]]] = None
    original_confs: Optional[List[float]] = None
    original_cls_ids: Optional[List[int]] = None

    result_boxes: Optional[List[List[float]]] = None
    result_confs: Optional[List[float]] = None
    result_cls_ids: Optional[List[int]] = None

    runtime_seconds: Optional[float] = None
    output_image_path: Optional[str] = None
    visualization_path: Optional[str] = None

    def to_rows(self) -> List[Dict[str, Any]]:
        """
        Generates a list of dictionaries, one for each original instance.
        If multiple instances exist, each gets its unique 'instance_confidence'
        AND its unique mapped 'result_confidence'.
        """
        rows = []

        # 1. Determine which original confidences to use as the base for rows
        confs_to_process = (
            self.original_confs
            if (self.original_confs and len(self.original_confs) > 0)
            else [self.original_confidence]
        )

        # 2. Handle the "No Detections" case
        if not confs_to_process or confs_to_process[0] is None:
            rows.append(
                self._create_row(instance_id=None, inst_conf=None, res_conf=None)
            )
            return rows

        # 3. Generate one row per animal, passing the mapped result confidence
        for i, conf in enumerate(confs_to_process):
            # Try to get the specific result confidence for this instance index
            # This relies on the spatial mapping logic we added to analyzer.py
            mapped_res_conf = None
            if self.result_confs and i < len(self.result_confs):
                mapped_res_conf = self.result_confs[i]

            rows.append(
                self._create_row(
                    instance_id=i, inst_conf=conf, res_conf=mapped_res_conf
                )
            )

        return rows

    def _create_row(
        self,
        instance_id: Optional[int],
        inst_conf: Optional[float],
        res_conf: Optional[float] = None,  # Added specific result confidence parameter
    ) -> Dict[str, Any]:
        """Helper to build the dictionary for a single row with mapped instance values."""

        # Determine the final result confidence for this row
        # Priority: 1. Passed mapped res_conf, 2. Global self.result_confidence, 3. 0.0
        final_res_conf = 0.0
        if res_conf is not None:
            final_res_conf = res_conf
        elif self.result_confidence is not None:
            # Fallback for single-instance legacy runs
            final_res_conf = self.result_confidence

        return {
            "experiment_type": self.experiment_type,
            "mask_mode": self.mask_mode,
            "image_path": self.image_path,
            "instance_id": instance_id if instance_id is not None else "N/A",
            "original_count": self.original_count,
            "result_count": self.result_count,
            "instance_confidence": (
                round(float(inst_conf), 4) if inst_conf is not None else 0.0
            ),
            "result_confidence": (round(float(final_res_conf), 4)),
            "outcome": self.outcome,
            "success": self.success,
            "inpainting_model": self.inpainting_model,
            "positive_prompt": self.positive_prompt or "",
            "runtime_seconds": (
                round(float(self.runtime_seconds), 3)
                if self.runtime_seconds is not None
                else 0.0
            ),
        }
