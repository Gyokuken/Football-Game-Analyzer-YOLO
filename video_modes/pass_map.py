import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from collections import defaultdict, deque
from sports.common.team import TeamClassifier
from video_modes.team_classification import get_crops
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from video_modes.radar import render_radar
import cv2
from video_modes.radar import COLORS as RADAR_COLORS

# Map: Team 0 -> RADAR_COLORS[0], Team 1 -> RADAR_COLORS[1], Referee -> RADAR_COLORS[3]
PASS_MAP_COLORS = [RADAR_COLORS[0], RADAR_COLORS[1], RADAR_COLORS[3]]

PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
BALL_DETECTION_MODEL_PATH = 'data/football-ball-detection.pt'
PITCH_DETECTION_MODEL_PATH = 'data/football-pitch-detection.pt'
PLAYER_CLASS_ID = 2
BALL_CLASS_ID = 0
STRIDE = 60
CONFIG = SoccerPitchConfiguration()
COLORS = ['#FF1493', '#00BFFF', '#FFD700']  # Team 0, Team 1, Referee
REFEREE_CLASS_ID = 3
BALL_HISTORY_LEN = 3
MIN_BALL_SPEED = 5  # pixels/frame
PASS_DISTANCE_THRESHOLD = 50

def run_pass_map(source_video_path: str, device: str):
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device)
    ball_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device)
    pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='Collecting crops for team classifier'):
        detections = sv.Detections.from_ultralytics(player_model(frame, imgsz=1280, verbose=False)[0])
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    ball_positions = deque(maxlen=BALL_HISTORY_LEN)
    player_positions = defaultdict(list)
    player_team_map = dict()
    pass_counts = defaultdict(lambda: defaultdict(int))
    last_possessor_id = None
    last_possessor_team = None
    last_possessor_ball_dist = None
    pass_cooldown = 0

    for frame in frame_generator:
        pitch_result = pitch_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

        player_result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(player_result)

        tracked = tracker.update_with_detections(detections)
        players = tracked[tracked.class_id == PLAYER_CLASS_ID]
        referees = tracked[tracked.class_id == REFEREE_CLASS_ID]

        if len(players) == 0:
            continue

        crops = get_crops(frame, players)
        player_team_ids = team_classifier.predict(crops)
        player_centroids = players.get_anchors_coordinates(anchor=sv.Position.CENTER)
        player_ids = players.tracker_id

        for idx, pid in enumerate(player_ids):
            player_positions[pid].append(player_centroids[idx])
            player_team_map[pid] = player_team_ids[idx]

        ball_result = ball_model(frame, imgsz=1280, verbose=False)[0]
        ball_detections = sv.Detections.from_ultralytics(ball_result)
        balls = ball_detections[ball_detections.class_id == BALL_CLASS_ID]
        if len(balls) == 0:
            continue
        
        ball_centroid = balls.get_anchors_coordinates(anchor=sv.Position.CENTER)[0]
        ball_positions.append(ball_centroid)

        if len(player_team_ids) != len(player_ids):
            continue

        # Find current possessor
        dists = np.linalg.norm(player_centroids - ball_centroid, axis=1)
        possessor_idx = np.argmin(dists)
        possessor_id = player_ids[possessor_idx]
        possessor_team = player_team_ids[possessor_idx]
        possessor_ball_dist = dists[possessor_idx]

        # Only count a pass if:
        # - possessor changed
        # - both previous and current possessor were close to the ball
        # - cooldown is 0
        if (
            last_possessor_id is not None
            and possessor_id != last_possessor_id
            and possessor_ball_dist < PASS_DISTANCE_THRESHOLD
            and last_possessor_ball_dist is not None
            and last_possessor_ball_dist < PASS_DISTANCE_THRESHOLD
            and pass_cooldown == 0
        ):
            print(f"PASS DETECTED: {last_possessor_id} (team {last_possessor_team}) -> {possessor_id} (team {possessor_team})")
            pass_counts[last_possessor_id][possessor_id] += 1
            pass_cooldown = 5  # frames

        if pass_cooldown > 0:
            pass_cooldown -= 1

        last_possessor_id = possessor_id
        last_possessor_team = possessor_team
        last_possessor_ball_dist = possessor_ball_dist

        # Merge players and referees for radar rendering
        detections = sv.Detections.merge([players, referees])
        color_lookup = np.concatenate([
            player_team_ids,  # 0 or 1 for teams
            np.full(len(referees), 2)  # 2 for referees
        ])

        radar_img = render_radar(detections, keypoints, color_lookup, colors=PASS_MAP_COLORS)
        radar_img = add_legend(radar_img, COLORS[:2] + ['#FFD700'])
        team_pass_total = {tid: 0 for tid in [0, 1]}
        
        for from_pid, to_dict in pass_counts.items():
            from_team = player_team_map.get(from_pid, None)
            if from_team is not None:
                team_pass_total[from_team] += sum(to_dict.values())

        yield frame, radar_img, team_pass_total

def add_legend(radar_img, team_colors):
    h, w, _ = radar_img.shape
    y = 30
    for tid, color in enumerate(team_colors):
        rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        if tid < 2:
            label = f'Team {tid}'
        else:
            label = 'Referee'
        cv2.circle(radar_img, (30, y), 10, rgb, -1)
        cv2.putText(radar_img, label, (50, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        y += 30
    return radar_img
