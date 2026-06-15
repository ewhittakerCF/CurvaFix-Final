# Purpose 
This GitHub repository contains scripts and modules that can be used to analyze data from biomechanical test videos. It measures and plots symphyseal gapping between two markers across the pubic symphysis throughout the duration of cyclical downward loading. It also enables comparison of  symphyseal gapping associated with different trials (e.g., different fixation methods). The code is written in Python.

# Installation

Start by installing Git on your machine. The following links are for Windows, Mac, and Linux, in order.  
```
https://git-scm.com/install/windows
```
```
https://git-scm.com/install/mac
```
```
https://git-scm.com/install/linux
```

After installing Git, clone this repository by running the following in the terminal:

```
cd <desired parent directory here>
git clone https://github.com/izstolzoff/CurvaFix.git
```

Next, install [Miniconda](https://www.anaconda.com/docs/getting-started/miniconda/install/overview), a Python package manager. After installing Miniconda, create and activate a new environment by running the following in the terminal: 

```
conda env create -f environment.yml
conda activate curvafix 
```

To use the environment you just created, run the following in the terminal. It should print a list of environments from which you can select curvafix.  
```
Python: Select Interpreter
```

Finally, install [VS Code](https://code.visualstudio.com/download), a code editor. Select the appropriate version for your machine. Once installed, install the Python Extension. While you can use any text editor to edit the files, VS Code is an intuitive user interface.

You're now ready to run the scripts!

# Usage

## constants.py
- Should be updated for each new set of video files that you are processing
- Includes data directory, video time segments, and marker color ranges.
- See within-file comments for details on updating constants.

## Analysis Scripts
Files 0-3 are typically run in order:

- `0_run_settling_analysis.py` produces visualizations to observe "settling" behavior at the start of a test. It plots the running standard deviations of the cycle max, mean, and min displacements over time.
- `1_process_videos.py` runs the video processor file to output raw marker displacements over time to CSV and png files. Follow instructions in the docstring to select the region of interest. 
- `2_run_stats.py` reads in the raw data saved from `1_process_videos.py` and computes symphyseal gapping relative to baseline. It saves time series data and plots that depict average and standard deviation displacements (max, mean, and min per cycle) over the cycles with processed video data.
- `3_create_comparison_plots.py` enables the user to plot displacement versus cycle for two different trials (e.g., to compare the CurvaFix Implant to a plate and screws.

More details for running these scripts are included in each file's docstring.

## Helper Files
- `raw_data_utils.py` includes helper methods for labeling and processing data.
- `video_data_manager.py` defines a class that used to find video files and define configurations associated with those video files for processing.
  - IMPORTANT: all videos should be named with "[number]k" cycles in the filename, separated using `_`s (e.g., `0k.mp4`, `curvafix_0k.mp4`, `curvafix_0k_trial1.mp4`.
- `video_processor.py` defines a class with methods to process the videos into raw marker distance data.

