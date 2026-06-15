import os

from constants import DATA_DIR, DEFAULT_VIDEO_START_END, FIRST_VIDEO_START_END


class VideoDataManager:
    """Manages the discovery and parsing of cycle data from video files."""

    def __init__(self, data_dir=DATA_DIR, first_file_start_end=FIRST_VIDEO_START_END):
        """
        Initialize the manager with directory paths and timing defaults.

        Args:
            data_dir (str): Path of directory containing videos.
            first_file_start_end (tuple): Start/end times for the 0k (initial) video.
        """
        self.data_dir = data_dir
        self.first_file_start_end = first_file_start_end

    def find_data_files(self):
        """
        Find individual .MP4 files within the specified data_dir, 
        excluding calibration files.

        Returns:
            list: Full paths to the discovered video files.
        """
        if not os.path.isdir(self.data_dir):
            raise FileNotFoundError(f"Data directory not found: {self.data_dir}")

        data_files = []
        for file in os.listdir(self.data_dir):
            if file.upper().endswith('.MP4') and 'calibration' not in file.lower():
                data_files.append(os.path.join(self.data_dir, file))
        return sorted(data_files)

    @staticmethod
    def extract_cycle_num(filename):
        """
        Extract cycle number from filename (e.g., '1k' -> 1000).

        Args:
            filename (str): Name or path of video file.
            
        Returns:
            int: Cycle number extracted from filename.
        """
        base_name = os.path.basename(filename)
        name_without_ext = os.path.splitext(base_name)[0]
        parts = name_without_ext.split('_')
        
        for part in parts:
            if part.lower().endswith('k'):
                try:
                    return int(part[:-1]) * 1000
                except ValueError:
                    continue
        
        raise ValueError(
            f"Could not extract cycle number from filename: {filename}. "
            "Expected format like '0k_part1_frontal.MP4' or '1k_frontal.MP4'."
        )

    def create_data_file_dict(self):
        """
        Create a dictionary mapping video filenames to cycle numbers and timing.

        Returns:
            dict: Mapping of {path: {'cycle_num': int, 'start_end': tuple}}
        """
        data_files = self.find_data_files()
        data_file_dict = {}

        for file in data_files:
            cycle_num = self.extract_cycle_num(file)
            
            # Use specific start/end for the first file (0k), otherwise default
            start_end = (
                self.first_file_start_end if cycle_num == 0 
                else DEFAULT_VIDEO_START_END
            )
            
            data_file_dict[file] = {
                'cycle_num': cycle_num, 
                'start_end': start_end
            }
            
            if cycle_num == 0: # add first second of video as baseline
                data_file_dict[f'{file}_baseline'] = {
                    'cycle_num': 'baseline',
                    'start_end': (0, 1)
                }
        
        return data_file_dict


if __name__ == "__main__":
    # Example Usage:
    manager = VideoDataManager()

    # Get the structured dictionary
    results = manager.create_data_file_dict()

    for path, info in results.items():
        print(f"File: {os.path.basename(path)} | Cycle: {info['cycle_num']} | Timing: {info['start_end']}")
