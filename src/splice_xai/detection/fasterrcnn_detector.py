import logging
import argparse
from typing import Optional, Dict, Any

import numpy as np
import torch
import torchvision
from torchvision.models.detection import (
    fasterrcnn_resnet50_fpn_v2,
    FasterRCNN_ResNet50_FPN_V2_Weights,
)

from splice_xai.core.results import ObjectDetectionResult

logger = logging.getLogger(__name__)


def _normalize_device(device: Optional[str]) -> str:
    if device in {None, "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device if torch.cuda.is_available() or device == "cpu" else "cpu"


class FasterRCNNDetector:
    def __init__(
        self, model_path: str, device: Optional[str] = None, num_classes: int = 1
    ):
        self.device = torch.device(_normalize_device(device))
        total_classes = num_classes + 1

        # 1. Initialize with DEFAULT weights first to fill the 'missing' backbone keys
        # Then we replace the head with your custom number of classes
        weights = FasterRCNN_ResNet50_FPN_V2_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn_v2(weights=weights)

        # Adjust the box predictor head to match your trained num_classes
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, total_classes
            )
        )

        try:
            torch.serialization.add_safe_globals([argparse.Namespace])
            checkpoint = torch.load(
                model_path, map_location=self.device, weights_only=False
            )

            # 2. Extract the weights from your specific key 'model_state_dict'
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]
            elif isinstance(checkpoint, dict) and "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            # Load the weights into the model
            self.model.load_state_dict(state_dict)
            logger.info(
                f"Successfully loaded weights from 'model_state_dict' in {model_path}"
            )

        except Exception as e:
            logger.error(f"Failed to load model weights: {e}")
            raise

        self.model.to(self.device)
        self.model.eval()

        self.class_names: Dict[int, str] = {
            i: f"class_{i}" for i in range(total_classes)
        }
        self.class_names[0] = "background"

    def detect(self, image: np.ndarray) -> Dict[str, torch.Tensor]:
        # Faster R-CNN expects float tensor [C, H, W] in [0, 1]
        img_tensor = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        img_tensor = img_tensor.unsqueeze(0).to(self.device)

        with torch.inference_mode():
            predictions = self.model(img_tensor)
        return predictions[0]

    def get_top_detection(
        self, image: np.ndarray, conf_threshold: float = 0.4
    ) -> ObjectDetectionResult:
        prediction = self.detect(image)
        boxes = prediction["boxes"].detach().cpu().numpy()
        scores = prediction["scores"].detach().cpu().numpy()
        labels = prediction["labels"].detach().cpu().numpy().astype(int)

        mask = scores >= conf_threshold
        if not np.any(mask):
            return ObjectDetectionResult()

        top_idx = int(np.argmax(scores))

        return ObjectDetectionResult(
            bbox=boxes[top_idx].tolist(),
            class_id=int(labels[top_idx]),
            confidence=float(scores[top_idx]),
            full_results={
                "all_boxes": boxes[mask].tolist(),
                "all_confs": scores[mask].tolist(),
                "all_cls_ids": labels[mask].tolist(),
            },
        )
