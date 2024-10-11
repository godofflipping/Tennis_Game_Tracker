import cv2
import sys
import utils
import numpy as np
from utils import (distance_meter_to_pixel, distance_pixel_to_meter,
                   get_bbox_center, get_distance, get_foot_position,
                   get_bbox_height, get_closest_keypoint, get_xy_distance)
sys.path.append('../')


class GameReconstructor:
    def __init__(self, frame):
        width, _, channels = frame.shape[:3]
        self.rectangle_width = width // 2
        self.rectangle_height = width
        self.channels = channels
        self.padding_court = int(self.rectangle_width * 0.1)

        self.set_canvas_background_box_position()
        self.set_mini_court_position()
        self.set_court_keypoints()
        self.set_court_lines()

    def convert_meter_to_pixel(self, meter_distance):
        return distance_meter_to_pixel(
            meter_distance, utils.DOUBLE_LINE_WIDTH, self.court_width
        )

    def set_mini_court_position(self):
        self.court_start_x = self.start_x + self.padding_court
        self.court_start_y = self.start_y + self.padding_court
        self.court_end_x = self.end_x - self.padding_court
        self.court_end_y = self.end_y - self.padding_court
        self.court_width = self.court_end_x - self.court_start_x

    def set_canvas_background_box_position(self):
        self.end_x = self.rectangle_width
        self.end_y = self.rectangle_height
        self.start_x = self.end_x - self.rectangle_width
        self.start_y = self.end_y - self.rectangle_height

    def set_court_keypoints(self):
        keypoints = [0] * 28
        # point 0
        keypoints[0] = int(self.court_start_x)
        keypoints[1] = int(self.court_start_y)
        # point 1
        keypoints[2] = int(self.court_end_x)
        keypoints[3] = int(self.court_start_y)
        # point 2
        keypoints[4] = int(self.court_start_x)
        keypoints[5] = self.court_start_y + self.convert_meter_to_pixel(
                                            utils.HALF_COURT_LINE_HEIGHT * 2)
        # point 3
        keypoints[6] = keypoints[0] + self.court_width
        keypoints[7] = keypoints[5]
        # #point 4
        keypoints[8] = keypoints[0] + self.convert_meter_to_pixel(
                                      utils.DOUBLE_ALLY_DIFFERENCE)
        keypoints[9] = keypoints[1]
        # #point 5
        keypoints[10] = keypoints[4] + self.convert_meter_to_pixel(
                                       utils.DOUBLE_ALLY_DIFFERENCE)
        keypoints[11] = keypoints[5]
        # #point 6
        keypoints[12] = keypoints[2] - self.convert_meter_to_pixel(
                                       utils.DOUBLE_ALLY_DIFFERENCE)
        keypoints[13] = keypoints[3]
        # #point 7
        keypoints[14] = keypoints[6] - self.convert_meter_to_pixel(
                                       utils.DOUBLE_ALLY_DIFFERENCE)
        keypoints[15] = keypoints[7]
        # #point 8
        keypoints[16] = keypoints[8]
        keypoints[17] = keypoints[9] + self.convert_meter_to_pixel(
                                       utils.NO_MANS_LAND_HEIGHT)
        # # #point 9
        keypoints[18] = keypoints[16] + self.convert_meter_to_pixel(
                                        utils.SINGLE_LINE_WIDTH)
        keypoints[19] = keypoints[17]
        # #point 10
        keypoints[20] = keypoints[10]
        keypoints[21] = keypoints[11] - self.convert_meter_to_pixel(
                                        utils.NO_MANS_LAND_HEIGHT)
        # # #point 11
        keypoints[22] = keypoints[20] + self.convert_meter_to_pixel(
                                        utils.SINGLE_LINE_WIDTH)
        keypoints[23] = keypoints[21]
        # # #point 12
        keypoints[24] = int((keypoints[16] + keypoints[18]) / 2)
        keypoints[25] = keypoints[17]
        # # #point 13
        keypoints[26] = int((keypoints[20] + keypoints[22]) / 2)
        keypoints[27] = keypoints[21]

        self.keypoints = keypoints

    def set_court_lines(self):
        self.lines = [(0, 2), (4, 5), (6, 7), (1, 3),
                      (0, 1), (8, 9), (10, 11), (10, 11), (2, 3)]

    def reconstruct_background(self):
        shapes = np.zeros((self.rectangle_height,
                           self.rectangle_width,
                           self.channels), np.uint8)
        cv2.rectangle(shapes, (self.start_x, self.start_y),
                      (self.end_x, self.end_y), (255, 255, 255), cv2.FILLED)
        return shapes

    def reconstruct_court(self, frame):
        for i in range(0, len(self.keypoints), 2):
            x = int(self.keypoints[i])
            y = int(self.keypoints[i + 1])
            cv2.circle(frame, (x, y), 7, (0, 0, 255), cv2.FILLED)

        # draw Lines
        for line in self.lines:
            start_point = (int(self.keypoints[line[0] * 2]),
                           int(self.keypoints[line[0] * 2 + 1]))
            end_point = (int(self.keypoints[line[1] * 2]),
                         int(self.keypoints[line[1] * 2 + 1]))
            cv2.line(frame, start_point, end_point, (0, 0, 0), 2)

        # Draw net
        net_start_point = (self.keypoints[0],
                           int((self.keypoints[1] + self.keypoints[5]) / 2))
        net_end_point = (self.keypoints[2],
                         int((self.keypoints[1] + self.keypoints[5]) / 2))
        cv2.line(frame, net_start_point, net_end_point, (255, 0, 0), 2)

        return frame

    def reconstruct_court_map(self, frames):
        output_frames = []
        for frame in frames:
            frame = self.reconstruct_background()
            frame = self.reconstruct_court(frame)
            output_frames.append(frame)
        return output_frames

    def get_start_point(self):
        return (self.court_start_x, self.court_start_y)

    def get_court_width(self):
        return self.court_width

    def get_keypoints(self):
        return self.keypoints

    def get_court_coordinates(self, object_position,
                              closest_keypoint,
                              closest_keypoint_index,
                              player_height_in_pixels,
                              player_height_in_meters):

        distance_x_pixels, distance_y_pixels = get_xy_distance(
            object_position,
            closest_keypoint
        )

        # Conver pixel distance to meters
        distance_from_keypoint_x_meters = distance_pixel_to_meter(
            distance_x_pixels,
            player_height_in_meters,
            player_height_in_pixels
        )
        distance_from_keypoint_y_meters = distance_pixel_to_meter(
            distance_y_pixels,
            player_height_in_meters,
            player_height_in_pixels
        )

        # Convert to mini court coordinates
        mini_court_x_distance_pixels = self.convert_meter_to_pixel(
            distance_from_keypoint_x_meters
        )
        mini_court_y_distance_pixels = self.convert_meter_to_pixel(
            distance_from_keypoint_y_meters
        )
        closest_mini_coourt_keypoint = (
            self.keypoints[closest_keypoint_index * 2],
            self.keypoints[closest_keypoint_index * 2+1]
        )

        mini_court_player_position = (
            closest_mini_coourt_keypoint[0] + mini_court_x_distance_pixels,
            closest_mini_coourt_keypoint[1] + mini_court_y_distance_pixels
        )

        return mini_court_player_position

    def reconstruct_bboxes(self, player_bboxes, ball_bboxes, keypoints):
        output_player_bboxes = []
        output_ball_bboxes = []
        max_player_bbox_height = 0

        for i, player_bbox in enumerate(player_bboxes):
            ball_box = ball_bboxes[i][1]
            ball_position = get_bbox_center(ball_box)
            closest_player_id = min(
                player_bbox.keys(), key=lambda x:
                get_distance(ball_position, get_bbox_center(player_bbox[x]))
            )

            output_player_bboxes_dict = {}
            for player_id, bbox in player_bbox.items():
                foot_position = get_foot_position(bbox)

                # Get The closest keypoint in pixels
                closest_keypoint_index = get_closest_keypoint(
                    foot_position, keypoints, [0, 2, 12, 13]
                )
                closest_keypoint = (
                    keypoints[closest_keypoint_index * 2],
                    keypoints[closest_keypoint_index * 2 + 1]
                )

                # Get Player height in pixels
                max_player_height_in_pixels = max(
                    get_bbox_height(player_bboxes[i][player_id]),
                    max_player_bbox_height
                )

                court_player_position = self.get_court_coordinates(
                    foot_position,
                    closest_keypoint,
                    closest_keypoint_index,
                    max_player_height_in_pixels,
                    utils.PLAYER_HEIGHT
                )

                output_player_bboxes_dict[player_id] = court_player_position

                if closest_player_id == player_id:
                    # Get The closest keypoint in pixels
                    closest_keypoint_index = get_closest_keypoint(
                        ball_position,
                        keypoints,
                        [0, 2, 12, 13]
                    )
                    closest_keypoint = (
                        keypoints[closest_keypoint_index * 2],
                        keypoints[closest_keypoint_index * 2 + 1]
                    )

                    court_player_position = self.get_court_coordinates(
                        ball_position,
                        closest_keypoint,
                        closest_keypoint_index,
                        max_player_height_in_pixels,
                        utils.PLAYER_HEIGHT
                    )
                    output_ball_bboxes.append({1: court_player_position})
            output_player_bboxes.append(output_player_bboxes_dict)

        return output_player_bboxes, output_ball_bboxes

    def reconstruct_court_points(self, frames, positions, color):
        for i, frame in enumerate(frames):
            for _, position in positions[i].items():
                x, y = position
                x = int(x)
                y = int(y)
                cv2.circle(frame, (x, y), 5, color, cv2.FILLED)
        return frames
