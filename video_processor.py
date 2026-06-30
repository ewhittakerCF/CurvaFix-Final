"""
Shared video processing utilities for marker extraction and displacement export.
"""

import os
import shutil

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

        self.marker_dist_df = pd.DataFrame(
            {
                "time_s": self.time,

                "posB_x_mm": self.posB_mm[:, 0],
                "posB_y_mm": self.posB_mm[:, 1],
                "posR_x_mm": self.posR_mm[:, 0],
                "posR_y_mm": self.posR_mm[:, 1],

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
            self.plot_marker_positions()

        shutil.copy("constants.py", self.data_dir)

    def plot_marker_positions(self):
        """Plot x and y positions over time for each marker."""

        # Blue marker position plot
        plt.figure(figsize=(6, 4))
        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["posB_x_mm"],
            color=CURVAFIX_COLORS[0],
            label="Blue marker x-position",
        )

        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["posB_y_mm"],
            color=CURVAFIX_COLORS[1],
            label="Blue marker y-position",
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Blue Marker Position (mm)")
        plt.legend()
        plt.tight_layout()

        blue_save_path = os.path.join(
            self.raw_data_plots_dir,
            f"posB_xy_vs_time_cycle_{self.cycle_num}.png"
        )

        plt.savefig(blue_save_path, bbox_inches="tight", dpi=300)
        plt.close()

        # Red marker position plot
        plt.figure(figsize=(6, 4))
        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["posR_x_mm"],
            color=CURVAFIX_COLORS[0],
            label="Red marker x-position",
        )

        plt.plot(
            self.marker_dist_df["time_s"],
            self.marker_dist_df["posR_y_mm"],
            color=CURVAFIX_COLORS[1],
            label="Red marker y-position",
        )

        plt.xlabel("Time (s)")
        plt.ylabel("Red Marker Position (mm)")
        plt.legend()
        plt.tight_layout()

        red_save_path = os.path.join(
            self.raw_data_plots_dir,
            f"posR_xy_vs_time_cycle_{self.cycle_num}.png"
        )

        plt.savefig(red_save_path, bbox_inches="tight", dpi=300)
        plt.close()


if __name__ == "__main__":
    video_processor = VideoProcessor()
    video_processor.run()
