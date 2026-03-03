import os
import shutil
import csv
from ultralytics import YOLO


def detect_and_log_consolidated(
    model_path, source_dir, save_dir, csv_path, conf_threshold=0.40
):
    # 1. Load Model
    model = YOLO(model_path)

    # 2. Prepare Directories
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 3. Run Inference
    results = model.predict(source=source_dir, conf=conf_threshold, stream=True)

    # 4. Process and Save
    with open(csv_path, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Header: filename, total count, and a single string of all box data
        writer.writerow(["filename", "detection_count", "all_detections_data"])

        print(f"Processing... saving matches to {save_dir}")
        saved_count = 0

        for result in results:
            if len(result.boxes) > 0:
                original_path = result.path
                filename = os.path.basename(original_path)

                # Create a list of strings for all detections in this specific image
                image_detections = []
                for box in result.boxes:
                    coords = [round(x, 2) for x in box.xyxy[0].tolist()]
                    conf = float(box.conf[0])
                    cls_name = model.names[int(box.cls[0])]

                    # Format individual detection info as a readable string
                    det_info = f"[Class: {cls_name}, Conf: {conf:.2f}, Box: {coords}]"
                    image_detections.append(det_info)

                # Join all detections with a semicolon so they stay in one CSV cell
                detections_string = " | ".join(image_detections)

                # Write one row for the whole image
                writer.writerow([filename, len(result.boxes), detections_string])

                # Copy the clean original image
                shutil.copy(original_path, os.path.join(save_dir, filename))
                saved_count += 1

    print(f"Done! {saved_count} images processed. Results saved in {csv_path}")


# --- Configuration ---
if __name__ == "__main__":
    # Update these paths!
    MY_MODEL = "data/models/seal_yolov9.pt"
    INPUT_IMG_DIR = "data/images/origin_test/seal_test"
    OUTPUT_IMG_DIR = "data/images/filtered/seal_filtered"
    LOG_FILE = "data/images/filtered/seal_detections.csv"

    detect_and_log_consolidated(
        model_path=MY_MODEL,
        source_dir=INPUT_IMG_DIR,
        save_dir=OUTPUT_IMG_DIR,
        csv_path=LOG_FILE,
        conf_threshold=0.40,
    )
