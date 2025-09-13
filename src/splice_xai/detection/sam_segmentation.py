import logging
from pathlib import Path
from typing import Optional

import numpy as np
import requests
import torch
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry

logger = logging.getLogger(__name__)


def _normalize_device(device: Optional[str]) -> str:
    if device in {None, "auto"}:
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


class SAMSegmentor:
    def __init__(
        self,
        checkpoint_path: str = "data/models/sam_vit_b_01ec64.pth",
        checkpoint_url: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.device = _normalize_device(device)
        self.checkpoint_path = checkpoint_path
        self.checkpoint_url = checkpoint_url

        if self.checkpoint_url:
            self._ensure_checkpoint(self.checkpoint_url)

        self._init_sam()

    def _ensure_checkpoint(self, url: str) -> None:
        path = Path(self.checkpoint_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            logger.info("SAM checkpoint already present")
            return

        logger.info(f"Downloading SAM checkpoint to {path} ...")
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with path.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1_048_576):
                        if chunk:
                            f.write(chunk)
        except Exception as e:
            if path.exists():
                path.unlink(missing_ok=True)
            raise RuntimeError(f"SAM checkpoint download failed: {e}")

    def _init_sam(self) -> None:
        logger.info(f"Loading SAM (vit_b) on {self.device} ...")
        sam = sam_model_registry["vit_b"](checkpoint=self.checkpoint_path)
        sam.to(self.device)
        sam.eval()
        self.predictor = SamPredictor(sam)
        logger.info("SAM ready")

    def generate_mask(
        self, image: np.ndarray, boxes: np.ndarray, mode: str = "union"
    ) -> Optional[Image.Image]:
        """
        Generate a mask from one or more XYXY boxes using SAM.
        - image: HWC RGB uint8
        - boxes: (N,4) float XYXY in absolute pixel coords
        """
        if image is None or image.ndim != 3:
            return None
        if boxes is None or len(boxes) == 0:
            return None

        # Ensure correct dtype/layout for SAM
        img_rgb = np.ascontiguousarray(image).astype(np.uint8)

        self.predictor.set_image(img_rgb)

        H, W = img_rgb.shape[:2]
        boxes = np.asarray(boxes, dtype=np.float32)

        with torch.inference_mode():
            if mode == "union":
                union_mask = np.zeros((H, W), dtype=np.uint8)
                for box in boxes:
                    masks, _, _ = self.predictor.predict(
                        point_coords=None,
                        point_labels=None,
                        box=box,
                        multimask_output=False,
                    )
                    if masks is None or len(masks) == 0:
                        continue
                    union_mask = np.maximum(union_mask, (masks[0].astype(np.uint8) * 255))

                if union_mask.max() == 0:
                    return None
                return Image.fromarray(union_mask, mode="L")

            elif mode == "top1":
                box = boxes[0]
                masks, _, _ = self.predictor.predict(
                    point_coords=None,
                    point_labels=None,
                    box=box,
                    multimask_output=False,
                )
                if masks is None or len(masks) == 0:
                    return None
                return Image.fromarray((masks[0].astype(np.uint8) * 255), mode="L")

        return None
