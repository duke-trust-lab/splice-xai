import glob
import os
import pandas as pd


def calculate_metrics(df):
    """Calculates the average runtime_seconds with robust error handling."""

    # 1. Ensure numeric types for calculation
    df["runtime_seconds"] = pd.to_numeric(df["runtime_seconds"], errors="coerce")

    # 2. Drop rows with missing crucial data so they don't skew the average
    df = df.dropna(subset=["runtime_seconds"])

    # 3. Guard against completely empty DataFrames
    if df.empty:
        return {"avg_runtime_seconds": 0.0}

    # Calculate average runtime_seconds across all records in this folder
    avg_runtime = df["runtime_seconds"].mean()

    if pd.isna(avg_runtime):
        avg_runtime = 0.0

    return {"avg_runtime_seconds": avg_runtime}


def process_experiments():
    # Set this to the parent directory where your  folders live
    root_dir = "final_replace"
    all_exp_data = []

    if not os.path.exists(root_dir):
        print(f"Error: Folder '{root_dir}' not found.")
        return

    # Filter folders that specifically start with ''
    experiments = [
        d
        for d in os.listdir(root_dir)
        if os.path.isdir(os.path.join(root_dir, d)) and d.startswith("")
    ]

    for exp in experiments:
        exp_folder_path = os.path.join(root_dir, exp)
        csv_files = glob.glob(os.path.join(exp_folder_path, "*.csv"))

        # Filter out any summary files we might have created in previous runs
        csv_files = [f for f in csv_files if "metrics_summary" not in f]

        if csv_files:
            print(f"Processing Folder: {exp} ({len(csv_files)} CSVs found)")

            try:
                temp_dfs = []
                for f in csv_files:
                    try:
                        # Only read the column we need to save memory and time
                        temp_df = pd.read_csv(f, usecols=["runtime_seconds"])
                        if not temp_df.empty:
                            temp_dfs.append(temp_df)
                    except ValueError:
                        print(
                            f"  -> Warning: 'runtime_seconds' column missing in {os.path.basename(f)}"
                        )
                    except Exception as csv_err:
                        print(f"  -> Warning: Could not read file {f}: {csv_err}")

                if not temp_dfs:
                    print(f"Skipping {exp}: No valid 'runtime_seconds' data found.")
                    continue

                combined_df = pd.concat(temp_dfs, ignore_index=True)

                # Calculate average
                metrics = calculate_metrics(combined_df)
                metrics["folder_name"] = exp
                all_exp_data.append(metrics)

            except Exception as e:
                print(f"  !! Error processing {exp}: {e}")
        else:
            print(f"Skipping {exp}: No CSV files found.")

    # Create and print the master summary table
    if all_exp_data:
        master_df = pd.DataFrame(all_exp_data)

        # Reorder to put folder_name first
        cols = ["folder_name", "avg_runtime_seconds"]
        master_df = master_df[cols]

        # Save to CSV
        master_path = os.path.join(root_dir, "runtime_summary_report.csv")
        master_df.to_csv(master_path, index=False)

        # Print the final table directly to the console
        print("\n" + "=" * 45)
        print("                 SUMMARY TABLE                 ")
        print("=" * 45)
        print(master_df.to_string(index=False))
        print("=" * 45)
        print(f"SUCCESS: Report saved to {master_path}")
    else:
        print("\nNo data was processed. Check your folder structure.")


if __name__ == "__main__":
    process_experiments()
