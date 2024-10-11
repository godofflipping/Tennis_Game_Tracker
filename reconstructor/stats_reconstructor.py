import sys
import utils
import pandas as pd
from copy import deepcopy
from utils import get_distance, distance_pixel_to_meter
sys.path.append('../')


class StatsReconstructor:
    def __init__(self, ball_hit_frames, num_frames, court_len_pixels):
        self.ball_hit_frames = ball_hit_frames
        self.num_frames = num_frames
        self.court_len_pixels = court_len_pixels
        self.players_stats = [{
            'frame': 0,
            'player_1_hits': 0,
            'player_1_total_b_speed': 0,
            'player_1_last_b_speed': 0,
            'player_1_total_speed': 0,
            'player_1_last_speed': 0,
            'player_2_hits': 0,
            'player_2_total_b_speed': 0,
            'player_2_last_b_speed': 0,
            'player_2_total_speed': 0,
            'player_2_last_speed': 0
        }]

    def get_full_stats(self, player_points, ball_points):
        for i in range(len(self.ball_hit_frames) - 1):
            start_frame = self.ball_hit_frames[i]
            end_frame = self.ball_hit_frames[i + 1]
            ball_time = (end_frame - start_frame) / 24

            # Get distance covered by the ball
            distance_ball_pixels = get_distance(
                ball_points[start_frame][1],
                ball_points[end_frame][1]
            )
            distance_ball_meters = distance_pixel_to_meter(
                distance_ball_pixels,
                utils.DOUBLE_LINE_WIDTH,
                self.court_len_pixels
            )

            # Speed of the ball from m/s in km/h
            speed_of_ball_shot = distance_ball_meters / ball_time * 3.6

            # player who the ball
            player_positions = player_points[start_frame]
            player_shot_ball = min(
                player_positions.keys(), key=lambda player_id:
                get_distance(player_positions[player_id],
                             ball_points[start_frame][1])
            )

            # opponent player speed
            opponent_player_id = 1 if player_shot_ball == 2 else 2
            distance_opponent_pixels = get_distance(
                player_points[start_frame][opponent_player_id],
                player_points[end_frame][opponent_player_id]
            )
            distance_opponent_meters = distance_pixel_to_meter(
                distance_opponent_pixels,
                utils.DOUBLE_LINE_WIDTH,
                self.court_len_pixels
            )

            speed_of_opponent = distance_opponent_meters / ball_time * 3.6

            current_player_stats = deepcopy(self.players_stats[-1])
            current_player_stats['frame_num'] = start_frame
            current_player_stats[
                f'player_{player_shot_ball}_hits'
            ] += 1
            current_player_stats[
                f'player_{player_shot_ball}_total_b_speed'
            ] += speed_of_ball_shot
            current_player_stats[
                f'player_{player_shot_ball}_last_b_speed'
            ] = speed_of_ball_shot
            current_player_stats[
                f'player_{opponent_player_id}_total_speed'
            ] += speed_of_opponent
            current_player_stats[
                f'player_{opponent_player_id}_last_speed'
            ] = speed_of_opponent

            self.players_stats.append(current_player_stats)

        player_stats_df = pd.DataFrame(self.players_stats)
        frames_df = pd.DataFrame(
            {'frame_num': list(range(self.num_frames))}
        )
        player_stats_df = pd.merge(
            frames_df,
            player_stats_df,
            on='frame_num',
            how='left'
        )
        player_stats_df = player_stats_df.ffill()

        player_stats_df['player_1_average_shot_speed'] = player_stats_df[
            'player_1_total_b_speed'] / player_stats_df['player_1_hits']
        player_stats_df['player_2_average_shot_speed'] = player_stats_df[
            'player_2_total_b_speed'] / player_stats_df['player_2_hits']
        player_stats_df['player_1_average_player_speed'] = player_stats_df[
            'player_1_total_speed'] / player_stats_df['player_1_hits']
        player_stats_df['player_2_average_player_speed'] = player_stats_df[
            'player_2_total_speed'] / player_stats_df['player_2_hits']

        return player_stats_df

    def save_stats(self, player_stats_df, path_to_save, index=False):
        player_stats_df.to_csv(path_to_save, index=index)
