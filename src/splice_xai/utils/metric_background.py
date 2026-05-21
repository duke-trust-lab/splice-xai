import pandas as pd
import os
import glob
import numpy as np


def calculate_metrics(df):
    """
    Calculates detection performance metrics based on original vs. results.
    """
    # 1. Identify types of rows
    # Original instances have numeric IDs; hallucinations start with 'new_'
    df["is_hallucination"] = (
        df["instance_id"].astype(str).str.startswith("new", na=False)
    )
    # We treat any non-hallucination row with a valid ID as an original instance
    df["is_original"] = ~df["is_hallucination"] & df["instance_id"].notna()

    # Calculate confidence delta
    df["conf_change"] = df["result_confidence"] - df["instance_confidence"]

    orig_df = df[df["is_original"]].copy()
    halluc_df = df[df["is_hallucination"]].copy()

    metrics = {}

    # --- METRIC GROUP 1: Original Object Preservation ---
    if not orig_df.empty:
        metrics["avg_conf_change_all"] = orig_df["conf_change"].mean()
        metrics["std_conf_change_all"] = orig_df["conf_change"].std()

        # Catastrophic Failure Rate (Reduction to 0)
        dropped_to_zero = len(orig_df[orig_df["result_confidence"] == 0])
        metrics["catastrophic_failure_rate_pct"] = (
            dropped_to_zero / len(orig_df)
        ) * 100

        # Retained Cases Only (Confidence > 0)
        retained_df = orig_df[orig_df["result_confidence"] > 0]
        if not retained_df.empty:
            metrics["avg_conf_change_retained"] = retained_df["conf_change"].mean()
            metrics["std_conf_change_retained"] = retained_df["conf_change"].std()
        else:
            metrics["avg_conf_change_retained"] = np.nan
            metrics["std_conf_change_retained"] = np.nan

        # Recall at 0.5 Threshold
        recalled = len(orig_df[orig_df["result_confidence"] >= 0.5])
        metrics["recall_at_0.5_threshold"] = recalled / len(orig_df)

    # --- METRIC GROUP 2: Hallucination Analysis ---
    # Group by image path to see how many hallucinations occur per file
    halluc_counts_per_image = df.groupby("image_path")["is_hallucination"].sum()
    metrics["avg_hallucinations_per_image"] = halluc_counts_per_image.mean()
    metrics["std_hallucinations_per_image"] = halluc_counts_per_image.std()
    metrics["avg_hallucination_confidence"] = (
        halluc_df["result_confidence"].mean() if not halluc_df.empty else 0
    )

    # --- METRIC GROUP 3: Efficiency & Outcomes ---
    # Average Runtime (Only count each image's runtime once)
    unique_image_data = df.drop_duplicates(subset=["image_path"])
    metrics["avg_runtime_seconds"] = unique_image_data["runtime_seconds"].mean()

    # Average Net Detection Change per image (result_count - original_count)
    metrics["avg_net_count_shift"] = (
        unique_image_data["result_count"] - unique_image_data["original_count"]
    ).mean()

    # Categorical Outcomes
    total_rows = len(df)
    if total_rows > 0:
        metrics["pct_original_preserved"] = (
            (df["outcome"] == "original_preserved").sum() / total_rows * 100
        )
        metrics["pct_hallucination"] = (
            (df["outcome"] == "hallucination").sum() / total_rows * 100
        )

    return metrics


def run_experiment_analysis(root_path):
    """
    Crawls directories, processes CSVs, and saves metrics.
    """
    # Finds all CSV files in subdirectories of final_background
    search_pattern = os.path.join(root_path, "**", "*.csv")
    all_files = glob.glob(search_pattern, recursive=True)

    for file_path in all_files:
        # Skip previously generated metrics files to avoid infinite loops
        if "metrics_summary" in file_path:
            continue

        print(f"Processing: {file_path}")
        try:
            df = pd.read_csv(file_path)

            # Ensure required columns exist
            required_cols = [
                "instance_id",
                "result_confidence",
                "instance_confidence",
                "image_path",
            ]
            if not all(col in df.columns for col in required_cols):
                print(f"Skipping {file_path}: Missing required columns.")
                continue

            results = calculate_metrics(df)

            # Convert dict to DataFrame and save
            metrics_df = pd.DataFrame([results])
            output_name = os.path.join(
                os.path.dirname(file_path), "metrics_summary.csv"
            )
            metrics_df.to_csv(output_name, index=False)
            print(f"Successfully saved metrics to {output_name}")

        except Exception as e:
            print(f"Failed to process {file_path}: {e}")


if __name__ == "__main__":
    # Point this to your root directory
    BASE_DIR = "final_background"
    run_experiment_analysis(BASE_DIR)
