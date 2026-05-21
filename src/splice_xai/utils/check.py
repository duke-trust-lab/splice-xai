import os

base_dir = "/Users/jiayizhou/Desktop/splice-xai/final_replace/replace_success/whale_yolo_sam_vessel_filtered"

if not os.path.exists(base_dir):
    print(f"❌ The directory itself does not exist: {base_dir}")
else:
    print(f"✅ Directory found. Scanning contents...")
    files = os.listdir(base_dir)
    if not files:
        print("Empty directory.")
    else:
        print("Files found in folder:")
        for f in files[:]:  # Print first 10 files
            print(f" - {f}")
