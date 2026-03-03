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
    mask_mode: str = "top1"  # New field

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
        """Generates a list of dictionaries, one for each original instance."""
        rows = []

        # If no instances were detected originally, return one row for the image
        if not self.original_confs or len(self.original_confs) == 0:
            rows.append(
                self._create_row(instance_id=None, inst_conf=self.original_confidence)
            )
            return rows

        # Create a row for every detected instance
        for i, conf in enumerate(self.original_confs):
            rows.append(self._create_row(instance_id=i, inst_conf=conf))

        return rows

    def _create_row(
        self, instance_id: Optional[int], inst_conf: Optional[float]
    ) -> Dict[str, Any]:
        """Helper to build the dictionary for a single row."""
        return {
            "experiment_type": self.experiment_type,
            "mask_mode": self.mask_mode,
            "image_path": self.image_path,
            "instance_id": instance_id,  # Link rows to specific animal index
            "original_count": self.original_count,
            "result_count": self.result_count,
            "instance_confidence": inst_conf,
            "result_confidence": self.result_confidence,  # Detection score after inpainting
            "outcome": self.outcome,
            "success": self.success,
            "inpainting_model": self.inpainting_model,
            "positive_prompt": self.positive_prompt,
            "runtime_seconds": self.runtime_seconds,
        }
