import sys
import cv2
import pickle
from ultralytics import YOLO
from utils import get_bbox_center, get_distance
from collections import Counter
sys.path.append('../')


class PlayerTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_saved=False, path_to_save=None):
        player_detections = []
        if read_from_saved and path_to_save is not None:
            with open(path_to_save, 'rb') as file:
                player_detections = pickle.load(file)
            return player_detections

        for frame in frames:
            player_dict = self.detect_frame(frame)
            player_detections.append(player_dict)

        if path_to_save is not None:
            with open(path_to_save, 'wb') as file:
                pickle.dump(player_detections, file)

        return player_detections

    def detect_frame(self, frame):
        # Only 1 image
        outputs = self.model.track(frame, persist=True)[0]
        dict_id_names = outputs.names

        player_dict = {}
        for box in outputs.boxes:
            track_id = int(box.id.tolist()[0])
            result = box.xyxy.tolist()[0]
            object_class_id = box.cls.tolist()[0]
            object_class_name = dict_id_names[object_class_id]
            if object_class_name == 'person':
                player_dict[track_id] = result

        return player_dict

    def filter_players(self, player_detections, keypoints, only_first=True):
        if only_first:
            chosen_players = self.choose_players(
                player_detections[0], keypoints)
        else:
            all_possible_players = list()
            for i in range(len(player_detections)):
                players = self.choose_players(player_detections[i], keypoints)
                all_possible_players.append(players[0])
                all_possible_players.append(players[1])
            counter = Counter(all_possible_players)
            players = counter.most_common(2)
            chosen_players = (players[0][0], players[1][0])

        filtered_detections = []
        for player_dict in player_detections:
            filtered_dict = {
                min(2, track_id): bbox
                for track_id, bbox in player_dict.items()
                if track_id in chosen_players
            }
            filtered_detections.append(filtered_dict)

        return filtered_detections

    def choose_players(self, player_dict, keypoints):
        track_distances = []
        for track_id, bbox in player_dict.items():
            player_center = get_bbox_center(bbox)

            min_distance = float('inf')
            for i in range(len(keypoints), 2):
                keypoint = (keypoints[i], keypoints[i + 1])
                distance = get_distance(player_center, keypoint)
                min_distance = min(min_distance, distance)
            track_distances.append((track_id, min_distance))

        track_distances.sort(key=lambda x: x[1])
        return (track_distances[0][0], track_distances[1][0])

    def draw_bboxes(self, frames, player_detections):
        output_frames = []
        for frame, player_dict in zip(frames, player_detections):
            for track_id, bbox in player_dict.items():
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.putText(frame, f"Player ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            output_frames.append(frame)

        return output_frames
