from utils import (read_video, save_video, draw_frame_numbers)
from trackers import PlayerTracker, TennisBallTracker, KeypointsDetector
from reconstructor import GameReconstructor, StatsReconstructor


def main():
    use_saved_detection = True
    path_to_input = 'input/video.mp4'
    path_to_output = 'output/output_video.avi'
    path_to_reconstruct = 'output/recostruct_video.avi'
    path_to_player_model = 'models/yolov8x'
    path_to_tennis_ball_model = 'models/yolov5_best.pt'
    path_to_keypoints_model = 'models/keypoints_weights.pth'
    player_save_path = 'saved_detections/player_detections.pkl'
    ball_save_path = 'saved_detections/ball_detections.pkl'
    path_to_save = 'output/game_stats.csv'

    # Get frames from the video
    input_frames = read_video(path_to_input)

    # Initialize all trackers and a detector
    player_tracker = PlayerTracker(path_to_player_model)
    ball_tracker = TennisBallTracker(path_to_tennis_ball_model)
    keypoints_detector = KeypointsDetector(path_to_keypoints_model)

    # Get all of the bboxes and keypoints from video frames
    player_detections = player_tracker.detect_frames(
        input_frames,
        read_from_saved=use_saved_detection,
        path_to_save=player_save_path
    )
    ball_detections = ball_tracker.detect_frames(
        input_frames,
        read_from_saved=use_saved_detection,
        path_to_save=ball_save_path
    )

    ball_detections = ball_tracker.interpolate_trajectory(ball_detections)
    ball_hit_frames = ball_tracker.get_ball_hits(ball_detections)

    keypoints = keypoints_detector.predict(input_frames[0])

    player_detections = player_tracker.filter_players(
        player_detections, keypoints, only_first=False
    )

    # Draw bboxes and other elements on frames
    output_frames = player_tracker.draw_bboxes(
        input_frames, player_detections
    )
    output_frames = ball_tracker.draw_bboxes(
        output_frames, ball_detections
    )
    output_frames = keypoints_detector.draw_keypoints_frames(
        output_frames, keypoints
    )
    output_frames = draw_frame_numbers(output_frames)

    game_reconstructor = GameReconstructor(input_frames[0])
    stats_reconstructor = StatsReconstructor(
        ball_hit_frames, len(input_frames),
        game_reconstructor.get_court_width()
    )

    reconstruct_frames = game_reconstructor.reconstruct_court_map(
        output_frames
    )
    player_points, ball_points = game_reconstructor.reconstruct_bboxes(
        player_detections, ball_detections, keypoints
    )
    reconstruct_frames = game_reconstructor.reconstruct_court_points(
        reconstruct_frames, player_points, (0, 0, 255)
    )
    reconstruct_frames = game_reconstructor.reconstruct_court_points(
        reconstruct_frames, ball_points, (0, 255, 0)
    )

    players_stats_df = stats_reconstructor.get_full_stats(
        player_points, ball_points
    )

    # Save videos and stats
    stats_reconstructor.save_stats(
        players_stats_df, path_to_save, index=False
    )
    save_video(reconstruct_frames, path_to_reconstruct)
    save_video(output_frames, path_to_output)


if __name__ == "__main__":
    main()
