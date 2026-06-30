import numpy as np

DATA_DIR = r"C:\Users\MayaSurve\Box\CurvaFix R&D\Biomechanical Testing\HMC Continuation\June Testing Videos"

# frame widths, in the plane perpendicular to the camera that includes each marker, in centimeters. Adjust with each new camera setup.
# note: the two should be equal when the red and blue markers are in the same plane (e.g., in the frontal configuration) and should be different when the markers are in a different plane (e.g., in the sagittal configuration).
FRAME_WIDTH_IN_CM_RED = 28.6
FRAME_WIDTH_IN_CM_BLUE = 28.6

DEFAULT_VIDEO_START_END = (0, 13) # default start and end times in seconds for videos. Typically left as-is.
FIRST_VIDEO_START_END = (0, 13) # default start and end times in seconds for first video (which may need to skip settling time)

# frontal
HSV_THRESHOLDS = {
    "blue": {
        "bound_low": (85, 50, 50),
        "bound_high": (110, 255, 255)
    },
    "red_region_low": { # note red appears at start and end of HSV scale
        "bound_low": (0, 50, 50),
        "bound_high": (5, 255, 255)
    },
    "red_region_high": {
        "bound_low": (173, 50, 50),
        "bound_high": (181, 255, 255)
    }
}

# # sagittal
# HSV_THRESHOLDS = {
#     "blue": {
#         "bound_low": np.array([95, 180, 150]),
#         "bound_high": np.array([110, 255, 255])
#     },
#     "red_region_low": { # note red appears at start and end of HSV scale
#         "bound_low": np.array([0, 50, 50]),
#         "bound_high": np.array([173, 50, 50])
#     },
#     "red_region_high": {
#         "bound_low": np.array([5, 255, 255]),
#         "bound_high": np.array([181, 255, 255])
#     }
# }

# stats-finding constants (these typically will not change)
APPROX_FRAME_RATE = 60  # Hz
PEAK_PROMINENCE = 0.05
PEAK_DISTANCE_SECONDS = 0.5

CURVAFIX_COLORS = [(108/255, 169/255, 225/255), # light blue
                   (140/255, 86/255, 140/255), # purple
                   (27/255, 69/255, 110/255)] # dark blue
