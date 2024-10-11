import cv2
import pickle
from ultralytics import YOLO
from utils import convert_to_dataframe, changes_detector


class TennisBallTracker:
    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect_frames(self, frames, read_from_saved=False, path_to_save=None):
        ball_detections = []
        if read_from_saved and path_to_save is not None:
            with open(path_to_save, 'rb') as file:
                ball_detections = pickle.load(file)
            return ball_detections

        for frame in frames:
            ball_dict = self.detect_frame(frame)
            ball_detections.append(ball_dict)

        if path_to_save is not None:
            with open(path_to_save, 'wb') as file:
                pickle.dump(ball_detections, file)

        return ball_detections

    def detect_frame(self, frame):
        # Only 1 image
        outputs = self.model.predict(frame, conf=0.15)[0]

        ball_dict = {}
        for box in outputs.boxes:
            result = box.xyxy.tolist()[0]
            ball_dict[1] = result

        return ball_dict

    def interpolate_trajectory(self, ball_detections):
        ball_detections_df = convert_to_dataframe(ball_detections)
        # interpolate with pandas method and fill
        # begining element with the earliest non-emtpy
        ball_detections_df = ball_detections_df.interpolate()
        ball_detections_df = ball_detections_df.bfill()

        ball_detections = [
            {1: position} for position in ball_detections_df
            .to_numpy().tolist()
        ]
        return ball_detections

    def get_ball_hits(self, ball_detections, window=5,
                      min_period=1, is_centered=False,
                      window_change_frames=25):
        ball_detections_df = convert_to_dataframe(ball_detections)
        ball_detections_df["y_avg_raw"] = (
            ball_detections_df["y1"] + ball_detections_df["y2"]
        ) / 2
        ball_detections_df["y_avg"] = (
            ball_detections_df["y_avg_raw"]
            .rolling(window=window, min_periods=min_period, center=is_centered)
            .mean()
        )

        buffer = int(window_change_frames * 1.1)
        ball_detections_df["y_delta"] = ball_detections_df["y_avg_raw"].diff()
        ball_detections_df["is_hit"] = 0

        for i in range(1, len(ball_detections_df) - buffer):
            negative_change, positive_change = changes_detector(
                ball_detections_df, 'y_delta', i, i + 1
            )

            if negative_change or positive_change:
                monotonicity_counter = 0

                for j in range(i + 1, i + buffer + 1):
                    neg_next_change, pos_next_change = changes_detector(
                        ball_detections_df, 'y_delta', i, j
                    )
                    if neg_next_change and negative_change:
                        monotonicity_counter += 1
                    elif pos_next_change and positive_change:
                        monotonicity_counter += 1

                if window_change_frames <= monotonicity_counter:
                    ball_detections_df["is_hit"].iloc[i] = 1

        return ball_detections_df[ball_detections_df["is_hit"] == 1].index \
                                                                    .tolist()

    def draw_bboxes(self, frames, ball_detections):
        output_frames = []
        for frame, ball_dict in zip(frames, ball_detections):
            for track_id, bbox in ball_dict.items():
                x1, y1, x2, y2 = bbox
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.putText(frame, f"Ball ID: {track_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            output_frames.append(frame)

        return output_frames
