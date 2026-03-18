import os
import shutil
import cv2
from ultralytics import YOLO


# --- Configuration ---
def run_seal_detection(model_path, source_dir, save_dir, conf_threshold=0.40):
    """
    Loads model, detects seals, draws bounding boxes, and saves to directory.
    """
    # 1. Load the model
    model = YOLO(model_path)

    # 2. Prepare output directory
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")

    # 3. Run Inference
    # stream=True handles large folders without crashing your RAM
    results = model.predict(source=source_dir, conf=conf_threshold, stream=True)

    print(f"Processing images from {source_dir}...")
    saved_count = 0

    for result in results:
        # Check if any objects (seals) were detected above the threshold
        if len(result.boxes) > 0:
            # .plot() draws the boxes and labels on the image array
            annotated_frame = result.plot()

            # Extract original filename to reuse it
            filename = os.path.basename(result.path)
            save_path = os.path.join(save_dir, filename)

            # Save the image with boxes drawn on it
            cv2.imwrite(save_path, annotated_frame)
            saved_count += 1

    print(f"Finished! Saved {saved_count} annotated images to: {save_dir}")


# --- Main Execution Block ---
if __name__ == "__main__":
    # Update these paths to match your environment
    MODEL_FILE = "data/models/seal_yolov9.pt"
    INPUT_FOLDER = "test_final_result/seal_sam_all_yolo"
    OUTPUT_FOLDER = "test_final_results_predict"

    run_seal_detection(
        model_path=MODEL_FILE,
        source_dir=INPUT_FOLDER,
        save_dir=OUTPUT_FOLDER,
        conf_threshold=0.40,
    )
