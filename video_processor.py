"""
Shared video processing utilities for marker extraction and displacement export.
"""

import os
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from matplotlib.ticker import MultipleLocator

from constants import (
    CURVAFIX_COLORS,
    DATA_DIR,
    FRAME_WIDTH_IN_CM_BLUE,
    FRAME_WIDTH_IN_CM_RED,
    HSV_THRESHOLDS,
)
from video_data_manager import VideoDataManager


CM_TO_MM = 10
MAX_SETTLING_ANALYSIS_SECONDS = 250


class VideoProcessor:
    def __init__(self, data_dir=DATA_DIR, video_files=None):
        self.data_dir = data_dir
        self.data_manager = VideoDataManager(data_dir=data_dir)
        self.video_files = video_files if video_files is not None else self.data_manager.create_data_file_dict()

        self.raw_data_dir = os.path.join(data_dir, "raw_marker_distance_data")
        self.raw_data_plots_dir = os.path.join(self.raw_data_dir, "plots")
        self._make_directories([self.raw_data_dir, self.raw_data_plots_dir])

        self.video_idx = None
        self.video_file = None
        self.video_config = None
        self.cycle_num = None

        self.x = None
        self.y = None
        self.w = None
        self.h = None

        self.posB_mm = None
        self.posR_mm = None
        self.time = None
        self.marker_dist_df = None

    @staticmethod
    def _make_directories(directories):
        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def correct_video_file_path(self):
        if self.video_file.endswith("_baseline"):
            self.video_file = self.video_file[:-len("_baseline")]
            self.cycle_num = "baseline"

    def select_roi(self):
        """Allow the user to select a region of interest (ROI) in the first frame."""
        self.correct_video_file_path()
        cap = cv2.VideoCapture(self.video_file)
        ret, first_frame = cap.read()
        cap.release()

        if not ret or first_frame is None:
            raise RuntimeError(f"Could not read first frame from video: {self.video_file}")

        scale = 0.25
        display = cv2.resize(first_frame, (0, 0), fx=scale, fy=scale)
        roi = cv2.selectROI("Select Crop Area", display, fromCenter=False, showCrosshair=True)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        if roi[2] == 0 or roi[3] == 0:
            raise ValueError("ROI selection cancelled or invalid. Please select a non-zero crop area.")

        self.x, self.y, self.w, self.h = [int(v / scale) for v in roi]

    @staticmethod
    def _get_pos(mask):
        """Return centroid coordinates for the largest contour in mask."""
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if cnts:
            moments = cv2.moments(max(cnts, key=cv2.contourArea))
            if moments["m00"] != 0:
                return (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"])
        return (np.nan, np.nan)

    @staticmethod
    def get_average_minimum_y_reference(
        positions_mm,
        time_s,
        minimum_distance_seconds=0.5,
        prominence_mm=0.05,
    ):
        """
        Calculate the average [x, y] marker position at local minimum-y points.

        In image coordinates, smaller y-values are higher in the image.

        Parameters
        ----------
        positions_mm : np.ndarray
            Nx2 array containing [x, y] marker positions in millimeters.

        time_s : np.ndarray
            Time value corresponding to each marker position.

        minimum_distance_seconds : float
            Minimum time between detected y minima.

        prominence_mm : float
            Minimum prominence required for a y minimum to be accepted.

        Returns
        -------
        reference_position : np.ndarray
            Average [x, y] position at the detected minimum-y points.

        minimum_indices : np.ndarray
            Indices of the detected minimum-y points.
        """

        if len(positions_mm) == 0:
            raise ValueError(
                "Cannot calculate a minimum-y reference from empty position data."
            )

        if len(positions_mm) != len(time_s):
            raise ValueError(
                "positions_mm and time_s must contain the same number of samples."
            )

        y_position = positions_mm[:, 1]

        if len(time_s) > 1:
            sample_interval = np.nanmedian(np.diff(time_s))

            if not np.isfinite(sample_interval) or sample_interval <= 0:
                raise ValueError(
                    "Cannot determine sample spacing from the time data."
                )

            samples_per_second = 1.0 / sample_interval
        else:
            samples_per_second = 1.0

        minimum_distance_samples = max(
            1,
            int(round(
                minimum_distance_seconds * samples_per_second
            )),
        )

        # Multiplying y by -1 converts local minima into peaks.
        minimum_indices, _ = find_peaks(
            -y_position,
            distance=minimum_distance_samples,
            prominence=prominence_mm,
        )

        if len(minimum_indices) == 0:
            raise RuntimeError(
                "No local minimum-y points were detected. "
                "Try lowering prominence_mm."
            )

        # Use the average x and y coordinates at the minimum-y frames.
        reference_position = np.nanmean(
            positions_mm[minimum_indices],
            axis=0,
        )

        return reference_position, minimum_indices
    
    def process_video_window(self, video_file, start_time, end_time, roi_coords):
        """
        Process one video over the requested time window and return marker positions.

        Args:
            video_file (str): Full path to a video file.
            start_time (float): Window start in seconds.
            end_time (float | None): Window end in seconds. If None, process to video end.
            roi_coords (tuple): ROI tuple (x, y, w, h) in full-resolution coordinates.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray]:
                (time_seconds, blue_marker_mm, red_marker_mm)
        """
        show_video = True

        cap = cv2.VideoCapture(video_file)
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_file}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            cap.release()
            raise RuntimeError(f"Invalid FPS ({fps}) for video: {video_file}")

        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_duration = frame_count / fps if frame_count > 0 else None

        start_time = max(0.0, float(start_time))
        if end_time is None:
            bounded_end_time = video_duration if video_duration is not None else start_time
        else:
            bounded_end_time = float(end_time)
            if video_duration is not None:
                bounded_end_time = min(bounded_end_time, video_duration)

        if bounded_end_time <= start_time:
            cap.release()
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

        start_frame = int(start_time * fps)
        end_frame = int(bounded_end_time * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

        mm_per_pixel_red = FRAME_WIDTH_IN_CM_RED * CM_TO_MM / frame_width
        mm_per_pixel_blue = FRAME_WIDTH_IN_CM_BLUE * CM_TO_MM / frame_width

        x, y, w, h = roi_coords
        pos_blue = []
        pos_red = []

        for _ in range(start_frame, end_frame):
            ret, frame = cap.read()
            if not ret:
                break

            cropped = frame[y:y + h, x:x + w]
            hsv = cv2.cvtColor(cropped, cv2.COLOR_BGR2HSV)

            blue_mask = cv2.inRange(
                hsv,
                HSV_THRESHOLDS["blue"]["bound_low"],
                HSV_THRESHOLDS["blue"]["bound_high"],
            )
            red_mask = cv2.inRange(
                hsv,
                HSV_THRESHOLDS["red_region_low"]["bound_low"],
                HSV_THRESHOLDS["red_region_low"]["bound_high"],
            ) | cv2.inRange(
                hsv,
                HSV_THRESHOLDS["red_region_high"]["bound_low"],
                HSV_THRESHOLDS["red_region_high"]["bound_high"],
            )

            cxB, cyB = self._get_pos(blue_mask)
            cxR, cyR = self._get_pos(red_mask)
            pos_blue.append([cxB, cyB])
            pos_red.append([cxR, cyR])
            if self.video_idx == 0:
                green_mask = cv2.merge([np.zeros_like(blue_mask), blue_mask, np.zeros_like(blue_mask)])
                red_mask_colored = cv2.merge([red_mask, np.zeros_like(red_mask), np.zeros_like(red_mask)])
                overlay = cv2.addWeighted(cropped, 0.7, green_mask, 0.3, 0)
                overlay = cv2.addWeighted(overlay, 1.0, red_mask_colored, 0.3, 0)
                
                if not np.isnan(cxB):
                    cv2.circle(overlay, (int(cxB), int(cyB)), 5, (255, 0, 0), -1)
                if not np.isnan(cxR):
                    cv2.circle(overlay, (int(cxR), int(cyR)), 5, (0, 0, 255), -1)

                if show_video:
                    cv2.imshow("Marker Overlay", overlay)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        cv2.destroyWindow("Marker Overlay")
                        cv2.waitKey(1)
                        show_video = False

        cap.release()
        cv2.destroyAllWindows()
        cv2.waitKey(1)

        pos_blue = np.array(pos_blue, dtype=float)
        pos_red = np.array(pos_red, dtype=float)
        if len(pos_blue) == 0 or len(pos_red) == 0:
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

        valid = ~np.isnan(pos_blue[:, 0]) & ~np.isnan(pos_red[:, 0])
        if not np.any(valid):
            return np.array([], dtype=float), np.array([], dtype=float), np.array([], dtype=float)

        blue_y_px = pos_blue[valid, 1]
        red_y_px = pos_red[valid, 1]

        blue_y_range_px = np.nanmax(blue_y_px) - np.nanmin(blue_y_px)
        red_y_range_px = np.nanmax(red_y_px) - np.nanmin(red_y_px)

        print(f"Frame width: {frame_width} px")
        print(f"Blue scale: {mm_per_pixel_blue:.6f} mm/px")
        print(f"Red scale: {mm_per_pixel_red:.6f} mm/px")

        print(f"Blue y travel: {blue_y_range_px:.2f} px")
        print(f"Red y travel: {red_y_range_px:.2f} px")

        print(
            f"Blue converted travel: "
            f"{blue_y_range_px * mm_per_pixel_blue:.3f} mm"
        )
        print(
            f"Red converted travel: "
            f"{red_y_range_px * mm_per_pixel_red:.3f} mm"
        )

        print(
            f"Expected pixel travel for 2.54 mm, blue: "
            f"{2.54 / mm_per_pixel_blue:.2f} px"
        )
        print(
            f"Expected pixel travel for 2.54 mm, red: "
            f"{2.54 / mm_per_pixel_red:.2f} px"
        )
        pos_blue = pos_blue[valid] * mm_per_pixel_blue
        pos_red = pos_red[valid] * mm_per_pixel_red
        time = np.arange(len(pos_blue), dtype=float) / fps + start_time

        return time, pos_blue, pos_red

    def process_video(self):
        """Process a single video file using configured cycle window."""
        self.correct_video_file_path()
        self.cycle_num = self.video_config["cycle_num"]

        roi_coords = (self.x, self.y, self.w, self.h)
        start_time, end_time = self.video_config["start_end"]
        time, pos_blue_mm, pos_red_mm = self.process_video_window(
            video_file=self.video_file,
            start_time=start_time,
            end_time=end_time,
            roi_coords=roi_coords,
        )

        if len(time) == 0:
            raise RuntimeError(f"No marker detections found for video: {self.video_file}")

        self.time = time
        self.posB_mm = pos_blue_mm
        self.posR_mm = pos_red_mm

    def save_marker_distances(self, marker_dist_df_save_path=None):
        marker_gap_vectors = self.posR_mm - self.posB_mm
        marker_dist = np.linalg.norm(marker_gap_vectors, axis=1)
        marker_dist_horiz = marker_gap_vectors[:, 0]
        marker_dist_vert = marker_gap_vectors[:, 1]

        if self.cycle_num == "baseline":
            # The baseline segment is only 1 second and should be approximately
            # stationary, so there may be no cyclic minima to detect.
            posB_reference = np.nanmean(self.posB_mm, axis=0)
            posR_reference = np.nanmean(self.posR_mm, axis=0)

            posB_minimum_indices = np.array([], dtype=int)
            posR_minimum_indices = np.array([], dtype=int)

        else:
            # For actual loading-cycle videos, use the average position at
            # detected local minimum-y points.
            posB_reference, posB_minimum_indices = (
            self.get_average_minimum_y_reference(
            positions_mm=self.posB_mm,
            time_s=self.time,
            minimum_distance_seconds=0.5,
            prominence_mm=0.05,
        )
    )

        posR_reference, posR_minimum_indices = (
        self.get_average_minimum_y_reference(
            positions_mm=self.posR_mm,
            time_s=self.time,
            minimum_distance_seconds=0.5,
            prominence_mm=0.05,
        )
    )

        # Calculate displacement relative to the chosen reference
        posB_displacement = self.posB_mm - posB_reference
        posR_displacement = self.posR_mm - posR_reference

        # Displacement relative to each marker's average minimum-y position.
        posB_displacement = self.posB_mm - posB_reference
        posR_displacement = self.posR_mm - posR_reference

        print(
            f"Blue reference: "
            f"x={posB_reference[0]:.3f} mm, "
            f"y={posB_reference[1]:.3f} mm; "
            f"detected minima={len(posB_minimum_indices)}"
        )

        print(
            f"Red reference: "
            f"x={posR_reference[0]:.3f} mm, "
            f"y={posR_reference[1]:.3f} mm; "
            f"detected minima={len(posR_minimum_indices)}"
        )

        self.marker_dist_df = pd.DataFrame(
            {
                "time_s": self.time,

                "posB_x_mm": self.posB_mm[:, 0],
                "posB_y_mm": self.posB_mm[:, 1],
                "posR_x_mm": self.posR_mm[:, 0],
                "posR_y_mm": self.posR_mm[:, 1],

                "dispB_x_mm": posB_displacement[:, 0],
                "dispB_y_mm": posB_displacement[:, 1],
                "dispR_x_mm": posR_displacement[:, 0],
                "dispR_y_mm": posR_displacement[:, 1],

                "dist_net_mm": marker_dist,
                "dist_x_mm": marker_dist_horiz,
                "dist_y_mm": marker_dist_vert,
            }
        )

        if marker_dist_df_save_path is None:
            marker_dist_df_save_path = os.path.join(self.raw_data_dir, f"raw_data_cycle_{self.cycle_num}.csv")
        self.marker_dist_df.to_csv(marker_dist_df_save_path, index=False)
        print(f"Finished cycle {self.cycle_num}. Saved CSV to: {marker_dist_df_save_path}.")

    def plot_raw_data(self):
        plt.figure(figsize=(6, 4))
        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["dist_net_mm"],
            color=CURVAFIX_COLORS[0],
            label="net marker distance",
        )
        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["dist_x_mm"],
            color=CURVAFIX_COLORS[1],
            label="x distance",
        )
        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["dist_y_mm"],
            color=CURVAFIX_COLORS[2],
            label="y distance",
        )
        plt.legend()
        plt.xlabel("Time (s)")
        plt.ylabel("Marker Distance (mm)")
        raw_data_save_path = os.path.join(self.raw_data_plots_dir, f"raw_data_cycle_{self.cycle_num}.png")
        plt.savefig(raw_data_save_path, bbox_inches="tight", dpi=300)
        plt.close()

    def _get_baseline_video_file(self):
        """Return the path of the 0k (baseline) video file."""
        for path in self.data_manager.find_data_files():
            try:
                if VideoDataManager.extract_cycle_num(path) == 0:
                    return path
            except ValueError:
                continue
        raise FileNotFoundError(f"No 0k baseline video found in: {self.data_dir}")

    def save_settling_raw_marker_distances(
        self,
        max_seconds=MAX_SETTLING_ANALYSIS_SECONDS,
    ):
        """Process the baseline video and save raw displacement data only.

        Returns:
            str: Path to saved raw displacement CSV.
        """
        baseline_file = self._get_baseline_video_file()

        self.video_file = baseline_file
        self.select_roi()
        roi_coords = (self.x, self.y, self.w, self.h)

        print(f"Processing baseline: {os.path.basename(baseline_file)}")
        self.video_idx = 0
        self.time, self.posB_mm, self.posR_mm = self.process_video_window(
            video_file=baseline_file,
            start_time=0,
            end_time=max_seconds,
            roi_coords=roi_coords,
        )

        if len(self.time) == 0:
            raise RuntimeError(f"No marker detections found in baseline video: {baseline_file}")

        settling_dir = os.path.join(self.data_dir, "settling_analysis")
        os.makedirs(settling_dir, exist_ok=True)
        filepath = os.path.join(settling_dir, 'settling_raw_marker_distances.csv')
        self.save_marker_distances(filepath)
        return filepath

    def run(self):
        if not self.video_files:
            raise RuntimeError(f"No videos found in DATA_DIR: {self.data_dir}")

        self.video_file = list(self.video_files.keys())[0]
        self.select_roi()

        for self.video_idx, (self.video_file, self.video_config) in enumerate(self.video_files.items()):
            print(f"Processing: {os.path.basename(self.video_file)}...")
            self.process_video()
            self.save_marker_distances()
            self.plot_raw_data()
            self.plot_marker_displacements()

        shutil.copy("constants.py", self.data_dir)

    def plot_marker_displacements(self):
        """Plot x and y displacement over time for each marker."""

        # Blue marker displacement plot
        plt.figure(figsize=(6, 4))

        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["dispB_x_mm"],
            color=CURVAFIX_COLORS[0],
            label="X displacement",
        )

        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["dispB_y_mm"],
            color=CURVAFIX_COLORS[1],
            label="Y displacement",
        )

        plt.axhline(
            y=0,
            linewidth=0.8,
            color="black",
            alpha=0.5,
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Blue Marker Displacement (mm)")
        plt.title(f"Blue Marker Displacement — Cycle {self.cycle_num}")
        plt.legend()

        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        blue_save_path = os.path.join(
            self.raw_data_plots_dir,
            f"dispB_xy_vs_time_cycle_{self.cycle_num}.png",
        )

        plt.savefig(
            blue_save_path,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        print(f"Saved blue marker displacement plot to: {blue_save_path}")

        # Red marker displacement plot
        plt.figure(figsize=(6, 4))

        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["dispR_x_mm"],
            color=CURVAFIX_COLORS[0],
            label="X displacement",
        )

        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["dispR_y_mm"],
            color=CURVAFIX_COLORS[1],
            label="Y displacement",
        )

        plt.axhline(
            y=0,
            linewidth=0.8,
            color="black",
            alpha=0.5,
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Red Marker Displacement (mm)")
        plt.title(f"Red Marker Displacement — Cycle {self.cycle_num}")
        plt.legend()

        ax = plt.gca()
        ax.yaxis.set_major_locator(MultipleLocator(0.25))

        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        red_save_path = os.path.join(
            self.raw_data_plots_dir,
            f"dispR_xy_vs_time_cycle_{self.cycle_num}.png",
        )

        plt.savefig(
            red_save_path,
            bbox_inches="tight",
            dpi=300,
        )
        plt.close()

        print(f"Saved red marker displacement plot to: {red_save_path}")


if __name__ == "__main__":
    video_processor = VideoProcessor()
    video_processor.run()
