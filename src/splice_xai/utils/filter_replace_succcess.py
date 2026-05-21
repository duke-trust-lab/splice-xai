import pandas as pd
import os

# 1. Define your paths (Updated to 'results.csv')
base_dir = "/Users/jiayizhou/Desktop/splice-xai/final_replace/replace_success/whale_yolo_box_vessel_filtered"
csv_input = os.path.join(base_dir, "results.csv")
csv_output = os.path.join(base_dir, "filtered_results.csv")

# 2. Load the CSV
if not os.path.exists(csv_input):
    print(f"❌ Error: Still can't find the file at {csv_input}")
else:
    df = pd.read_csv(csv_input)
    print(f"✅ Loaded {len(df)} rows from results.csv")

    # 3. Get a set of all files in that folder for fast lookup
    files_in_folder = set(os.listdir(base_dir))

    def check_viz_exists(image_path):
        # image_path is like 'data/images/.../DSC03795.JPG'
        # We need 'DSC03795'
        if pd.isna(image_path):
            return False

        base_name = os.path.splitext(os.path.basename(str(image_path)))[0]
        # Look for the specific pattern you mentioned: DSC03795_replace_viz.png
        target_file = f"{base_name}_replace_viz.png"

        return target_file in files_in_folder

    # 4. Filter the rows
    # This keeps only the rows where the viz file actually exists on your desktop
    filtered_df = df[df["image_path"].apply(check_viz_exists)]

    # 5. Save the new CSV
    filtered_df.to_csv(csv_output, index=False)

    print(f"---")
    print(f"Success!")
    print(f"Rows kept: {len(filtered_df)} (out of {len(df)})")
    print(f"Filtered file saved to: {csv_output}")
