IMG_DIR = 'camera_cal'
OUT_DIR = 'output_images'

SOBEL_KERNEL = 7

LMAG_THRESHOLD = (10000, 50000)
LDIR_THRESHOLD = (0.7, 1.1)

SMAG_THRESHOLD = (10000, 50000)
SDIR_THRESHOLD = (0.7, 1.1)

ROAD_REGION = [[0, 660],      # left, bottom
                [565, 450],    # left, top
                [715, 450],    # right, top
                [1280, 660]]  # right, bottom

REG_DIM = (6.3, 37.0) # dimensions of road region in meters
LANE_WIDTH = 3.6 # lane width in meters

# UNDISTORTED_RECT = (144, 720) # width, height
OVERHEAD_RECT = (1280, 720) # width, height
POLY_DEG=3
N = 3

SCALE = 1.0

IMSCALE = 1.0
FPS = 10
VID_NAME = 'outvideo.mp4'
