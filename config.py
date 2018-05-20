IMG_DIR = 'camera_cal'
OUT_DIR = 'output_images'

SOBEL_KERNEL = 7

LMAG_THRESHOLD = (10000, 50000)
LDIR_THRESHOLD = (0.7, 1.1)

SMAG_THRESHOLD = (10000, 50000)
SDIR_THRESHOLD = (0.7, 1.1)

ROAD_REGION = [[0, 660],      # bottom left
                [540, 450],    # top left
                [740, 450],    # top right
                [1280, 660]]  # bottom right

# READ_LENGTH = 
REAL_WIDTH = 3.6 # read lane width is 3.6m

UNDISTORTED_RECT = (144, 720) # width, height
LANE_WIDTH=75
POLY_DEG=3
