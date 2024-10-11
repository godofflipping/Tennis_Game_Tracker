from .video_utils import (read_video, save_video, draw_frame_numbers)
from .bbox_utils import (get_bbox_center, get_distance,
                         get_foot_position, get_bbox_height,
                         get_xy_distance, get_closest_keypoint)
from .conversion_utils import (distance_pixel_to_meter,
                               distance_meter_to_pixel,
                               convert_to_dataframe, changes_detector)

# Tennis court dimensions
# https://www.harrodsport.com/uploads/wysiwyg/img/doubles-tennis-court-dimensions-598x381.png
SINGLE_LINE_WIDTH = 8.23
DOUBLE_LINE_WIDTH = 10.97
HALF_COURT_LINE_HEIGHT = 11.88
SERVICE_LINE_WIDTH = 6.4
DOUBLE_ALLY_DIFFERENCE = 1.37
NO_MANS_LAND_HEIGHT = 5.48

PLAYER_HEIGHT = 1.89
