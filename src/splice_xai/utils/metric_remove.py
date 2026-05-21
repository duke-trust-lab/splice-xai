import pandas as pd
import os
import glob


def calculate_metrics(df):
    """Calculates the specific metrics requested by the user."""
    # Ensure numeric types for calculation
    df["instance_confidence"] = pd.to_numeric(
        df["instance_confidence"], errors="coerce"
    )
    df["result_confidence"] = pd.to_numeric(df["result_confidence"], errors="coerce")
    df["runtime_seconds"] = pd.to_numeric(df["runtime_seconds"], errors="coerce")

    # Metric 1: Average confidence change and std dev (across all instances)
    df["conf_change"] = df["result_confidence"] - df["instance_confidence"]
    avg_conf_change = df["conf_change"].mean()
    std_conf_change = df["conf_change"].std()

    # Metric 2: Percentage of instances reduced to 0 confidence
    # Using 0.001 to account for float precision
    is_zero = df["result_confidence"] <= 0.001
    percent_zero = (is_zero.sum() / len(df)) * 100

    # Metric 3: For those NOT reduced to 0, average change and std dev
    non_zero_df = df[~is_zero]
    if not non_zero_df.empty:
        avg_change_non_zero = non_zero_df["conf_change"].mean()
        std_change_non_zero = non_zero_df["conf_change"].std()
    else:
        avg_change_non_zero = 0
        std_change_non_zero = 0

    # Metric 4: Average runtime per image
    # Note: Using .first() because runtime is usually logged per image,
    # and we don't want to double-count it for every instance in that image.
    avg_runtime = df.groupby("image_path")["runtime_seconds"].first().mean()

    return {
        "avg_conf_change_all": avg_conf_change,
        "std_conf_change_all": std_conf_change,
        "percent_reduced_to_zero": percent_zero,
        "avg_change_nonzero_only": avg_change_non_zero,
        "std_change_nonzero_only": std_change_non_zero,
        "avg_runtime_per_image": avg_runtime,
    }


def process_experiments():
    root_dir = "final_remove"
    all_exp_data = []

    if not os.path.exists(root_dir):
        print(f"Error: Folder '{root_dir}' not found.")
        return

    # Get list of experiment folders
    experiments = [
        d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))
    ]

    print(f"Found {len(experiments)} potential experiment folders.\n")

    for exp in experiments:
        exp_folder_path = os.path.join(root_dir, exp)
        # Search for CSVs directly in final_remove/experiment/*.csv
        csv_files = glob.glob(os.path.join(exp_folder_path, "*.csv"))

        # Filter out any summary files we might have created in previous runs
        csv_files = [f for f in csv_files if "metrics_summary" not in f]

        if csv_files:
            print(f"Processing Experiment: {exp} ({len(csv_files)} CSVs found)")

            try:
                # Combine all CSVs found in this experiment folder
                temp_dfs = [pd.read_csv(f) for f in csv_files]
                combined_df = pd.concat(temp_dfs, ignore_index=True)

                # Calculate metrics
                metrics = calculate_metrics(combined_df)
                metrics["experiment_name"] = exp
                all_exp_data.append(metrics)

                # Save individual experiment metrics CSV
                res_df = pd.DataFrame([metrics])
                individual_save_path = os.path.join(
                    exp_folder_path, "metrics_summary.csv"
                )
                res_df.to_csv(individual_save_path, index=False)

            except Exception as e:
                print(f"  !! Error processing {exp}: {e}")
        else:
            print(f"Skipping {exp}: No CSV files found.")

    # Create the master summary table in final_remove/
    if all_exp_data:
        master_df = pd.DataFrame(all_exp_data)
        # Ensure experiment_name is the first column
        cols = ["experiment_name"] + [
            c for c in master_df.columns if c != "experiment_name"
        ]
        master_df = master_df[cols]

        master_path = os.path.join(root_dir, "all_experiments_master_report.csv")
        master_df.to_csv(master_path, index=False)

        print("\n" + "=" * 30)
        print(f"SUCCESS: Master report saved to {master_path}")
        print("=" * 30)
    else:
        print("\nNo data was processed. Check your folder structure.")


if __name__ == "__main__":
    process_experiments()
