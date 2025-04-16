"""A script to merge files from particular experiments."""

import os
import pandas as pd


base_dir = "data/iou_curves"

file_names = [
    "comparison_ranking_partial.csv",
    "comparison_score_avg.csv",
    "comparison_ranking_avg.csv",
    "comparison_score_partial.csv"
]


def main() -> None:  # TODO: add column with dataset split type

    # create an empty dict to store data to merge
    merged_data = {name: [] for name in file_names}

    # iterate through subdirectories and read each CSV file
    for subdir in os.listdir(base_dir):
        subdir_path = os.path.join(base_dir, subdir)
        if os.path.isdir(subdir_path):
            for file_name in file_names:
                file_path = os.path.join(subdir_path, file_name)
                if os.path.isfile(file_path):
                    df = pd.read_csv(file_path)
                    df["source_folder"] = subdir  # a columnt with source dir name
                    merged_data[file_name].append(df)

    # save merged files into target directory
    for file_name, dfs in merged_data.items():
        if dfs:
            merged_df = pd.concat(dfs, ignore_index=True)
            output_path = os.path.join(base_dir, f"merged_{file_name}")
            merged_df.to_csv(output_path, index=False)
            print(f"Saved: {output_path}")
        else:
            print(f"No data for: {file_name}")


if __name__ == "__main__":
    main()
