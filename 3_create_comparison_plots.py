"""
Create comparison plots for the displacement data.

Some user-defined constants are required. Instructions for using the script:
1. Set the VIDEO_DIR to the directory containing the video data.
2. Set the RUNS_TO_COMPARE dictionary to include the runs you want to compare.
3. Set the SAVE_DIR to the directory where you want to save the plots.
4. Run `python 3_create_comparison_plots.py` to execute the script.
"""
import os
import pandas as pd
import matplotlib.pyplot as plt

from constants import CURVAFIX_COLORS


# user-defined constants
VIDEO_DIR = r'C:\Users\MayaSurve\Box\CurvaFix R&D\Biomechanical Testing\HMC Continuation\September Testing\CurvaFix and Straight Screw\Double Leg Stance\Test 135'

RUNS_TO_COMPARE = { # directories should contain displacement_summary csvs
    'CurvaFix Implant': r'CurvaFix\Frontal\concat',
    'Traditional Plate': r'Plate\Frontal\concat',
}

SAVE_DIR = r'C:\Users\MayaSurve\Box\CurvaFix R&D\Biomechanical Testing\HMC Continuation\September Testing\CurvaFix and Straight Screw\Double Leg Stance\Test 65\comparison_plots'
os.makedirs(SAVE_DIR)


for direction in ["net", "x", "y"]:
    plt.figure(figsize=(6,4))
    for run_idx, (run, run_dir) in enumerate(RUNS_TO_COMPARE.items()):
        file_path = os.path.join(VIDEO_DIR, run_dir, "stats", f'displacement_summary_{direction}.csv')
        df = pd.read_csv(file_path)

        # Access columns directly for plotting
        plt.errorbar(df["cycle_num"], df['max_avg'], yerr=df['max_sd'], color=CURVAFIX_COLORS[run_idx], label=run)
    
    plt.xlabel('Cycle Number')
    plt.ylabel(f'{direction.capitalize()} Displacement (mm)')
    plt.ylim([-0.1, 2])
    plt.legend()
    save_path = os.path.join(SAVE_DIR, f'displacement_comparison_{direction}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    