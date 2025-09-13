import logging
from typing import Optional, Dict, Any

import numpy as np
import torch
from ultralytics import YOLO

from splice_xai.core.results import ObjectDetectionResult

logger = logging.getLogger(__name__)


def _normalize_device(device: Optional[str]) -> str:
    if device in {None, "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device in {"cpu", "cuda"}:
        return device
    # Fallback for odd strings like "cuda:0"
    return device if torch.cuda.is_available() or device == "cpu" else "cpu"


class YOLODetector:
    def __init__(self, model_path: str, device: Optional[str] = None):
        self.device = _normalize_device(device)
        self.model = YOLO(model_path)

        # Try half precision on CUDA for speed; safe no-op on CPU.
        try:
            if self.device.startswith("cuda"):
                self.model.to(self.device)
                # Some Ultralytics models expose .model.half()
                if hasattr(self.model, "model"):
                    self.model.model.half()  # type: ignore[attr-defined]
        except Exception as e:
            logger.warning(f"Half precision not enabled: {e}")

        raw_names = getattr(self.model.model, "names", None)
        if isinstance(raw_names, dict):
            self.class_names: Dict[int, str] = raw_names
        elif isinstance(raw_names, (list, tuple)):
            self.class_names = {i: n for i, n in enumerate(raw_names)}
        else:
            # Fallback to empty mapping; callers should handle None names.
            self.class_names = {}

        logger.info(f"YOLO model loaded on {self.device}")
        if self.class_names:
            logger.info(f"Classes: {list(self.class_names.values())}")

    def detect(self, image: np.ndarray, conf_threshold: float = 0.4) -> Any:
        # Ultralytics accepts numpy HWC RGB arrays
        with torch.inference_mode():
            results = self.model(
                image,
                conf=conf_threshold,
                verbose=False,
                device=self.device,
            )
        return results[0]

    def get_top_detection(
        self, image: np.ndarray, conf_threshold: float = 0.4
    ) -> ObjectDetectionResult:
        detection = self.detect(image, conf_threshold)

        # Defensive checks in case no boxes are returned
        boxes_obj = getattr(detection, "boxes", None)
        if boxes_obj is None:
            return ObjectDetectionResult()

        # Tensors to CPU numpy
        try:
            xyxy = boxes_obj.xyxy
            confs = boxes_obj.conf
            clses = boxes_obj.cls
        except Exception:
            return ObjectDetectionResult()

        if xyxy is None or getattr(xyxy, "shape", (0, 0))[0] == 0:
            return ObjectDetectionResult()

        boxes = xyxy.detach().cpu().numpy()
        confs_np = confs.detach().cpu().numpy()
        clses_np = clses.detach().cpu().numpy().astype(int)

        top_idx = int(confs_np.argmax())

        return ObjectDetectionResult(
            bbox=boxes[top_idx].tolist(),
            class_id=int(clses_np[top_idx]),
            confidence=float(confs_np[top_idx]),
            full_results={
                "all_boxes": boxes.tolist(),
                "all_confs": confs_np.tolist(),
                "all_cls_ids": clses_np.tolist(),
            },
        )
