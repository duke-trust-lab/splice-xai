import os
import shutil
import logging
from pathlib import Path

# Import your existing project modules
from splice_xai.detection.yolo_detector import YOLODetector
from splice_xai.detection.fasterrcnn_detector import FasterRCNNDetector

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def filter_single_instance_images(
    source_dir: str,
    output_dir: str,
    model_path: str,
    model_type: str = "yolo",
    threshold: float = 0.4,
    device: str = "cpu",
):
    src_path = Path(source_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {model_type} model...")
    if model_type.lower() == "yolo":
        detector = YOLODetector(model_path, device=device)
    else:
        detector = FasterRCNNDetector(model_path, device=device, num_classes=1)

    image_files = [
        p for p in src_path.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    total = len(image_files)
    logger.info(f"Processing {total} images...")

    count_copied = 0
    for idx, img_p in enumerate(image_files):
        # Manual progress print every 10 images
        if idx % 10 == 0:
            logger.info(f"Progress: {idx}/{total} images analyzed...")

        try:
            detection = detector.get_top_detection(str(img_p), conf_threshold=threshold)
            all_boxes = detection.full_results.get("all_boxes", [])

            if len(all_boxes) == 1:
                shutil.copy2(img_p, out_path / img_p.name)
                count_copied += 1

        except Exception as e:
            logger.error(f"Error processing {img_p.name}: {e}")

    logger.info(f"Done! Saved {count_copied} single-instance images to {output_dir}")


if __name__ == "__main__":
    # --- CONFIGURATION ---
    SOURCE = "data/images/filtered/penguin_filtered"
    DESTINATION = "data/images/single/penguin_filtered_single"
    MODEL = "data/models/penguin_yolov9.pt"
    TYPE = "yolo"
    THRESH = 0.4

    filter_single_instance_images(
        source_dir=SOURCE,
        output_dir=DESTINATION,
        model_path=MODEL,
        model_type=TYPE,
        threshold=THRESH,
    )
