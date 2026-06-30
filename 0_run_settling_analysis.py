"""
Settling analysis on baseline (0k) cycle.

Workflow:
1. Extract raw baseline displacement from the first 250s (or full video if shorter).
2. Compute cumulative SD of max/mean/min displacement over time.
3. Save analysis outputs under DATA_DIR/settling_analysis.

To run:
    python 0_run_settling_analysis.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from constants import (
    APPROX_FRAME_RATE,
    CURVAFIX_COLORS,
    DATA_DIR,
    PEAK_DISTANCE_SECONDS,
    PEAK_PROMINENCE,
)
from video_processor import VideoProcessor


def summarize_sd_over_time(distance, time):
    displacement = distance - np.nanmean(distance[:int(APPROX_FRAME_RATE)]) # baseline-corrected displacement
    peak_distance_frames = max(1, int(APPROX_FRAME_RATE * PEAK_DISTANCE_SECONDS))
    peaks, _ = find_peaks(displacement, distance=peak_distance_frames, prominence=PEAK_PROMINENCE)
    troughs, _ = find_peaks(-displacement, distance=peak_distance_frames, prominence=PEAK_PROMINENCE)

    peak_values = displacement[peaks]
    trough_values = displacement[troughs]

    cycle_means = []
    for i in range(len(peaks) - 1):
        segment = displacement[peaks[i]:peaks[i + 1]]
        if len(segment) > 0:
            cycle_means.append(np.nanmean(segment))
    cycle_means = np.array(cycle_means, dtype=float)

    sd_max = np.array([np.nanstd(peak_values[: i + 1]) for i in range(len(peak_values))])
    sd_min = np.array([np.nanstd(trough_values[: i + 1]) for i in range(len(trough_values))])
    sd_mean = np.array([np.nanstd(cycle_means[: i + 1]) for i in range(len(cycle_means))])

    return {
        "peak_times": time[peaks],
        "trough_times": time[troughs],
        "cycle_times": np.mean([time[peaks[:-1]], time[peaks[1:]]], axis=0) if len(peaks) > 1 else np.array([]),
        "sd_max": sd_max,
        "sd_min": sd_min,
        "sd_mean": sd_mean,
    }


def save_analysis_outputs(sd_data, settling_dir):
    plot_path = os.path.join(settling_dir, "SD_plot_frontal.png")
    plt.figure(figsize=(6, 4))
    plt.plot(sd_data["peak_times"], sd_data["sd_max"], label="max", color=CURVAFIX_COLORS[0])
    plt.plot(sd_data["cycle_times"], sd_data["sd_mean"], label="mean", color=CURVAFIX_COLORS[1])
    plt.plot(sd_data["trough_times"], sd_data["sd_min"], label="min", color=CURVAFIX_COLORS[2])
    plt.xlabel("Time (s)")
    plt.ylabel("Standard Deviation (mm)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches="tight", dpi=300)
    plt.close()

    summary_df = pd.DataFrame(
        {
            "peak_time_s": pd.Series(sd_data["peak_times"]),
            "max_sd_mm": pd.Series(sd_data["sd_max"]),
            "cycle_time_s": pd.Series(sd_data["cycle_times"]),
            "mean_sd_mm": pd.Series(sd_data["sd_mean"]),
            "trough_time_s": pd.Series(sd_data["trough_times"]),
            "min_sd_mm": pd.Series(sd_data["sd_min"]),
        }
    )
    summary_csv_path = os.path.join(settling_dir, "settling_sd_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)

    print(f"Saved SD plot to: {plot_path}")
    print(f"Saved SD summary CSV to: {summary_csv_path}")


def run_settling_analysis(data_dir=DATA_DIR):
    processor = VideoProcessor(data_dir=data_dir)
    raw_csv_path = processor.save_settling_raw_marker_distances()

    raw_df = pd.read_csv(raw_csv_path)
    if raw_df.empty:
        raise RuntimeError(f"No raw displacement data found in: {raw_csv_path}")

    time = raw_df["time_s"].to_numpy(dtype=float)
    distance = raw_df["dist_net_mm"].to_numpy(dtype=float)

    sd_data = summarize_sd_over_time(distance, time)
    settling_dir = os.path.dirname(raw_csv_path)
    save_analysis_outputs(sd_data, settling_dir)


if __name__ == "__main__":
    run_settling_analysis()
