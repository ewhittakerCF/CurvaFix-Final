"""
Shared helpers for reading processed raw marker-distance cycle CSV files.
"""

import os
import re

import pandas as pd


def list_raw_data_files(raw_data_dir):
    return sorted(
        f for f in os.listdir(raw_data_dir)
        if f.startswith("raw_data_cycle_") and f.endswith(".csv")
    )


def get_baseline_file(files):
    baseline_files = [f for f in files if "baseline" in f]
    if len(baseline_files) != 1:
        raise ValueError(f"Expected 1 baseline file, but found {len(baseline_files)}")
    return baseline_files[0]


def extract_cycle_num(file_name):
    match = re.search(r"raw_data_cycle_(\d+)", file_name)
    if not match:
        raise ValueError(f"Could not parse cycle number from file: {file_name}")
    return int(match.group(1))


def compute_baseline_values(raw_data_dir, baseline_file):
    baseline_df = pd.read_csv(os.path.join(raw_data_dir, baseline_file))
    return {
        "net": baseline_df["dist_net_mm"].mean(),
        "x": baseline_df["dist_x_mm"].mean(),
        "y": baseline_df["dist_y_mm"].mean(),
    }
