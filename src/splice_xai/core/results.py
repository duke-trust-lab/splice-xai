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
        return bool(boxes)  # True if list exists and non-empty


@dataclass
class CounterfactualResult:
    image: Optional[Image.Image] = None
    mask: Optional[Image.Image] = None
    success: bool = False
    outcome: str = "unsuccessful"

    experiment_type: str = ""  # 'remove' | 'replace' | 'background'
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

    original_boxes: Optional[List[List[float]]] = None
    original_confs: Optional[List[float]] = None
    original_cls_ids: Optional[List[int]] = None

    result_boxes: Optional[List[List[float]]] = None
    result_confs: Optional[List[float]] = None
    result_cls_ids: Optional[List[int]] = None

    runtime_seconds: Optional[float] = None

    output_image_path: Optional[str] = None
    visualization_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "experiment_type": self.experiment_type,
            "image_path": self.image_path,
            "mask_path": self.mask_path,
            "background_type": self.background_type,
            "target_label": self.target_label,
            "inpainting_model": self.inpainting_model,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "original_class_name": self.original_class_name,
            "result_class_name": self.result_class_name,  # standardized key
            "original_count": self.original_count,
            "result_count": self.result_count,
            "original_confidence": self.original_confidence,
            "result_confidence": self.result_confidence,
            "outcome": self.outcome,
            "success": self.success,
            "runtime_seconds": self.runtime_seconds,
        }
