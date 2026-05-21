import pandas as pd

# Load the filtered CSV
file_path = "/Users/jiayizhou/Desktop/splice-xai/final_replace/replace_success/whale_yolo_sam_vessel_filtered/filtered_results.csv"

try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    print(f"Error: File not found at {file_path}")
    exit()

# --- 1. Basic Setup ---
total_instances = len(df)
df["conf_drop"] = df["instance_confidence"] - df["result_confidence"]
near_zero_thresh = 0.001

# --- 2. Calculation Logic ---

# Success metrics
success_count = df["success"].sum()
success_rate = (success_count / total_instances) * 100

# Threshold counts
below_04 = df[df["result_confidence"] < 0.4].shape[0]
nearly_zero = df[df["result_confidence"] <= near_zero_thresh].shape[0]

# Stubborn detection analysis (The new metric)
# These are cases where the object was NOT fully removed
stubborn_df = df[df["result_confidence"] > near_zero_thresh]
if not stubborn_df.empty:
    avg_drop_stubborn = stubborn_df["conf_drop"].mean()
    avg_remaining_conf = stubborn_df["result_confidence"].mean()
else:
    avg_drop_stubborn = 0
    avg_remaining_conf = 0

# --- 3. Final Print Out ---

print("=" * 60)
print("             IMAGE INPAINTING ANALYSIS REPORT")
print("=" * 60)

print(f"## GENERAL STATISTICS")
print(f"Total Instances Processed : {total_instances}")
print(f"Overall Success Rate      : {success_rate:.2f}%")
print(f"  (Success = instances where target object confidence <0.4 after editing)")
print(f"Average Runtime           : {df['runtime_seconds'].mean():.3f}s")
print(f"Max Runtime               : {df['runtime_seconds'].max():.3f}s")

print(f"\n## DETECTION EFFECTIVENESS")
print(f"Confidence < 0.4          : {below_04} instances")
print(f"  (Objects reduced below the standard detection threshold)")
print(f"Confidence ≈ 0 (<= {near_zero_thresh})   : {nearly_zero} instances")
print(f"  (Objects effectively erased from the detector's view)")
print(f"Average Confidence Drop   : {df['conf_drop'].mean():.4f}")
print(f"  (The average amount of certainty the AI lost after editing)")

print(f"\n## RESIDUAL (STUBBORN) DETECTIONS")
print(f"Instances not near zero   : {len(stubborn_df)}")
print(f"  (Cases where a detection of > {near_zero_thresh} confidence remains)")
print(f"Avg Drop for Stubborn     : {avg_drop_stubborn:.4f}")
print(f"  (Even if not erased, how much did we weaken the detection?)")
print(f"Avg Remaining Confidence  : {avg_remaining_conf:.4f}")

print(f"\n## OUTCOME BREAKDOWN")
print(df["outcome"].value_counts().to_string())
print("=" * 60)
