import os
import shutil

# --- Configuration ---
source_dir = (
    "final_replace/whale_yolo_sam_vessel"  # Change this to your source directory
)
target_dir = "final_replace/whale_yolo_sam_vessel_filtered"  # Change this to your target directory
exclude_strings = [
    "DSC00429",
    "DSC01615",
    "DSC02239",
    "DSC02689",
    "DSC03560",
    "DSC04331",
    "DSC04565",
    "DSC09056",
    "DSC09070",
]

# Common image extensions to ensure we only move photos, not system files
valid_extensions = (".jpg", ".jpeg", ".png", ".arw", ".cr2", ".nef", ".dng")


def filter_out_images():
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

    count = 0
    for filename in os.listdir(source_dir):
        # Only process files with image extensions
        if filename.lower().endswith(valid_extensions):

            # The Logic Flip: Check if NONE of the exclude strings are in the filename
            if not any(s in filename for s in exclude_strings):
                source_path = os.path.join(source_dir, filename)
                target_path = os.path.join(target_dir, filename)

                shutil.copy2(source_path, target_path)
                count += 1

    print(f"Success! {count} images (excluding your list) were copied to {target_dir}")


if __name__ == "__main__":
    filter_out_images()
