"""
Process video data to extract marker positions and calculate displacements. Running this file should prompt you to crop the video aorund the markers as instructed in the test protocol. 
After selecting the ROI, hit 'enter'. Hit 'q' on your keyboard to close the mask overlay video. 

To run:
    python 1_process_videos.py
"""

from video_processor import VideoProcessor


if __name__ == "__main__":
    video_processor = VideoProcessor()
    video_processor.run()
    
