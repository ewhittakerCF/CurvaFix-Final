"""
Run statistics on the processed video data.

Some user-defined constants are required. Instructions for using the script:
1. Set the DATA_DIR constant to the path of your data directory.
2. Run `python 2_run_stats.py` to execute the script.
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from raw_data_utils import (
    compute_baseline_values,
    extract_cycle_num,
    get_baseline_file,
    list_raw_data_files,
)
from constants import (
    APPROX_FRAME_RATE,
    PEAK_PROMINENCE,
    PEAK_DISTANCE_SECONDS,
    CURVAFIX_COLORS
)


DATA_DIR = r"G:\.shortcut-targets-by-id\1N6eMHszDLGNamwgRzvzJqe-KKoNW98HP\CurvaFix, Inc\Videos (1)\CurvaFix 0-10k\CurvaFix 0-6k Frontal"


def summarize_displacement(displacement, approx_frame_rate):
    peak_distance_frames = max(1, int(approx_frame_rate * PEAK_DISTANCE_SECONDS))
    peaks, _ = find_peaks(
        displacement,
        distance=peak_distance_frames,
        prominence=PEAK_PROMINENCE,
    )

    if len(peaks) < 2:
        return {
            "max_avg": np.nan,
            "max_sd": np.nan,
            "mean_avg": np.nan,
            "mean_sd": np.nan,
            "min_avg": np.nan,
            "min_sd": np.nan,
        }

    cycle_stats = []
    for i in range(len(peaks) - 1):
        start_idx = peaks[i]
        end_idx = peaks[i + 1]
        single_cycle = displacement[start_idx:end_idx]

        if len(single_cycle) == 0:
            continue

        cycle_stats.append(
            {
                "mean": np.nanmean(single_cycle),
                "min": np.nanmin(single_cycle),
                "max": np.nanmax(single_cycle),
            }
        )

    if not cycle_stats:
        return {
            "max_avg": np.nan,
            "max_sd": np.nan,
            "mean_avg": np.nan,
            "mean_sd": np.nan,
            "min_avg": np.nan,
            "min_sd": np.nan,
        }

    cycle_stats_df = pd.DataFrame(cycle_stats)
    return {
        "max_avg": np.nanmean(cycle_stats_df["max"]),
        "max_sd": np.nanstd(cycle_stats_df["max"]),
        "mean_avg": np.nanmean(cycle_stats_df["mean"]),
        "mean_sd": np.nanstd(cycle_stats_df["mean"]),
        "min_avg": np.nanmean(cycle_stats_df["min"]),
        "min_sd": np.nanstd(cycle_stats_df["min"]),
    }


def save_summary_and_plot(summary_df, direction, stats_dir):
    summary_df = summary_df.sort_values(by="cycle_num")
    summary_df.to_csv(os.path.join(stats_dir, f"displacement_summary_{direction}.csv"), index=False)

    plt.figure(figsize=(6, 4))
    for stat_idx, stat in enumerate(["max", "mean", "min"]):
        plt.errorbar(
            summary_df["cycle_num"],
            summary_df[f"{stat}_avg"],
            yerr=summary_df[f"{stat}_sd"],
            color=CURVAFIX_COLORS[stat_idx],
            label=stat,
        )
    plt.ylim([-2, 2])
    plt.legend()
    plt.xlabel("Cycle Number")
    plt.ylabel(f"{direction.capitalize()} Displacement (mm)")
    stat_over_cycle_save_path = os.path.join(stats_dir, f"disp_{direction}_vs_cycle.png")
    plt.savefig(stat_over_cycle_save_path, bbox_inches="tight", dpi=300)
    plt.close()


def run_stats(approx_frame_rate=APPROX_FRAME_RATE):
    raw_data_dir = os.path.join(DATA_DIR, "raw_marker_distance_data")
    if not os.path.isdir(raw_data_dir):
        raise FileNotFoundError(f"Raw data directory not found: {raw_data_dir}")

    files = list_raw_data_files(raw_data_dir)
    if not files:
        raise FileNotFoundError(f"No raw data files found in: {raw_data_dir}")

    baseline_file = get_baseline_file(files)
    baseline_values = compute_baseline_values(raw_data_dir, baseline_file)

    summaries = {"net": [], "x": [], "y": []}

    for file_name in files:
        if file_name == baseline_file:
            continue

        cycle_num = extract_cycle_num(file_name)
        cycle_df = pd.read_csv(os.path.join(raw_data_dir, file_name))

        displacements = {
            "net": cycle_df["dist_net_mm"] - baseline_values["net"],
            "x": cycle_df["dist_x_mm"] - baseline_values["x"],
            "y": cycle_df["dist_y_mm"] - baseline_values["y"],
        }

        for direction, displacement in displacements.items():
            summary = summarize_displacement(displacement, approx_frame_rate)
            summary["cycle_num"] = cycle_num
            summaries[direction].append(summary)

    stats_dir = os.path.join(DATA_DIR, "stats")
    os.makedirs(stats_dir, exist_ok=True)
    
    for direction, rows in summaries.items():
        summary_df = pd.DataFrame(rows)
        if summary_df.empty:
            continue
        summary_df["cycle_num"] = summary_df["cycle_num"].astype(int)
        save_summary_and_plot(summary_df, direction, stats_dir)


if __name__ == "__main__":
    run_stats()
