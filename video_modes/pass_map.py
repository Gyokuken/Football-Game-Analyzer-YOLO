# import numpy as np
# import supervision as sv
# from ultralytics import YOLO
# from tqdm import tqdm
# from collections import defaultdict, deque
# from sports.common.team import TeamClassifier
# from video_modes.team_classification import get_crops
# from sports.configs.soccer import SoccerPitchConfiguration
# from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
# from video_modes.radar import render_radar
# import cv2
# from video_modes.radar import COLORS as RADAR_COLORS

# # Map: Team 0 -> RADAR_COLORS[0], Team 1 -> RADAR_COLORS[1], Referee -> RADAR_COLORS[3]
# PASS_MAP_COLORS = [RADAR_COLORS[0], RADAR_COLORS[1], RADAR_COLORS[3]]

# PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
# BALL_DETECTION_MODEL_PATH = 'data/football-ball-detection.pt'
# PITCH_DETECTION_MODEL_PATH = 'data/football-pitch-detection.pt'
# PLAYER_CLASS_ID = 2
# BALL_CLASS_ID = 0
# STRIDE = 60
# CONFIG = SoccerPitchConfiguration()
# COLORS = ['#FF1493', '#00BFFF', '#FFD700']  # Team 0, Team 1, Referee
# REFEREE_CLASS_ID = 3
# BALL_HISTORY_LEN = 3
# MIN_BALL_SPEED = 5  # pixels/frame
# PASS_DISTANCE_THRESHOLD = 50

# def run_pass_map(source_video_path: str, device: str):
#     import csv

#     player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device)
#     ball_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device)
#     pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device)

#     # Collect crops for team classification
#     frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
#     crops = []
#     for frame in tqdm(frame_generator, desc='Collecting crops for team classifier'):
#         detections = sv.Detections.from_ultralytics(player_model(frame, imgsz=1280, verbose=False)[0])
#         crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

#     team_classifier = TeamClassifier(device=device)
#     team_classifier.fit(crops)

#     # Setup for main video processing
#     frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
#     tracker = sv.ByteTrack(minimum_consecutive_frames=3)

#     ball_positions = deque(maxlen=BALL_HISTORY_LEN)
#     player_positions = defaultdict(list)
#     player_team_map = dict()
#     pass_counts = defaultdict(lambda: defaultdict(int))

#     last_possessor_id = None
#     last_possessor_team = None
#     last_possessor_ball_dist = None

#     ball_released_by = None
#     ball_released_team = None
#     ball_released_frame = None
#     pass_armed = False

#     BALL_VELOCITY_THRESHOLD = 2  # pixels/frame
#     PASS_DISTANCE_THRESHOLD = 10  # pixels
#     PASS_TIMEOUT = 20  # max frames after ball release

#     pass_log_path = "pass_log.csv"
#     frame_index = 0

#     with open(pass_log_path, mode="w", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow([
#             "frame_index",
#             "from_player_id",
#             "from_team",
#             "to_player_id",
#             "to_team",
#             "ball_velocity",
#             "from_dist",
#             "to_dist"
#         ])

#         for frame in frame_generator:
#             print(f"\n--- Frame {frame_index} ---")
#             frame_index += 1

#             pitch_result = pitch_model(frame, verbose=False)[0]
#             keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

#             player_result = player_model(frame, imgsz=1280, verbose=False)[0]
#             detections = sv.Detections.from_ultralytics(player_result)
#             tracked = tracker.update_with_detections(detections)

#             players = tracked[tracked.class_id == PLAYER_CLASS_ID]
#             referees = tracked[tracked.class_id == REFEREE_CLASS_ID]

#             if len(players) == 0:
#                 print("No players detected")
#                 continue

#             crops = get_crops(frame, players)
#             player_team_ids = team_classifier.predict(crops)
#             player_centroids = players.get_anchors_coordinates(anchor=sv.Position.CENTER)
#             player_ids = players.tracker_id

#             for idx, pid in enumerate(player_ids):
#                 player_positions[pid].append(player_centroids[idx])
#                 player_team_map[pid] = player_team_ids[idx]

#             ball_result = ball_model(frame, imgsz=1280, verbose=False)[0]
#             ball_detections = sv.Detections.from_ultralytics(ball_result)
#             balls = ball_detections[ball_detections.class_id == BALL_CLASS_ID]
#             if len(balls) == 0:
#                 print("No ball detected")
#                 continue

#             ball_centroid = balls.get_anchors_coordinates(anchor=sv.Position.CENTER)[0]
#             ball_positions.append(ball_centroid)

#             if len(ball_positions) < 2:
#                 print("Insufficient ball position history")
#                 continue

#             # Compute ball velocity
#             ball_velocity = np.linalg.norm(ball_positions[-1] - ball_positions[-2])
#             print(f"Ball velocity: {ball_velocity:.2f}")

#             # Find current closest player
#             dists = np.linalg.norm(player_centroids - ball_centroid, axis=1)
#             possessor_idx = np.argmin(dists)
#             possessor_id = player_ids[possessor_idx]
#             possessor_team = player_team_ids[possessor_idx]
#             possessor_ball_dist = dists[possessor_idx]

#             print(f"Closest player: ID {possessor_id}, Team {possessor_team}, Distance to ball: {possessor_ball_dist:.2f}")

#             # --- Pass Logic ---

#             if (
#                 last_possessor_id is not None
#                 and ball_velocity > BALL_VELOCITY_THRESHOLD
#                 and last_possessor_ball_dist is not None
#                 and last_possessor_ball_dist < PASS_DISTANCE_THRESHOLD
#             ):
#                 print(f"Ball released by {last_possessor_id} (Team {last_possessor_team})")
#                 ball_released_by = last_possessor_id
#                 ball_released_team = last_possessor_team
#                 ball_released_frame = frame_index
#                 pass_armed = True

#             if (
#                 pass_armed
#                 and ball_released_by != possessor_id
#                 and possessor_ball_dist < PASS_DISTANCE_THRESHOLD
#                 and (frame_index - ball_released_frame <= PASS_TIMEOUT)
#             ):
#                 print(f"✅ PASS DETECTED: {ball_released_by} (team {ball_released_team}) -> {possessor_id} (team {possessor_team})")
#                 pass_counts[ball_released_by][possessor_id] += 1

#                 writer.writerow([
#                     frame_index,
#                     ball_released_by,
#                     ball_released_team,
#                     possessor_id,
#                     possessor_team,
#                     round(ball_velocity, 2),
#                     round(last_possessor_ball_dist, 2),
#                     round(possessor_ball_dist, 2)
#                 ])
#                 pass_armed = False  # reset

#             elif pass_armed and frame_index - ball_released_frame > PASS_TIMEOUT:
#                 print(f"Pass timeout: no receiver within {PASS_TIMEOUT} frames")
#                 pass_armed = False

#             # # Debug for cases where no pass
#             # if not pass_armed:
#             #     print("No pass armed or detected")

#             # above line has been replaced with the below lines --------------

#             ball_travel_distance = 0
#             if len(ball_positions) >= 2:
#                 ball_travel_distance = np.linalg.norm(ball_positions[-1] - ball_positions[0])

#             # Calculate ball trajectory angle (if possible)
#             ball_direction_angle = None
#             if len(ball_positions) >= 3:
#                 vec1 = ball_positions[-2] - ball_positions[-3]
#                 vec2 = ball_positions[-1] - ball_positions[-2]
#                 if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
#                     cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
#                     ball_direction_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

#             # Only count passes if both players are on the same team and ball traveled enough distance
#             MIN_PASS_TRAVEL_DIST = 20  # pixels
#             MAX_PASS_ANGLE_CHANGE = 60 # degrees, to filter out rebounds

#             if (
#                 last_possessor_id is not None
#                 and ball_velocity > BALL_VELOCITY_THRESHOLD
#                 and last_possessor_ball_dist is not None
#                 and last_possessor_ball_dist < PASS_DISTANCE_THRESHOLD
#             ):
#                 ball_released_by = last_possessor_id
#                 ball_released_team = last_possessor_team
#                 ball_released_frame = frame_index
#                 ball_release_position = ball_positions[-2] if len(ball_positions) >= 2 else ball_positions[-1]
#                 pass_armed = True

#             if (
#                 pass_armed
#                 and ball_released_by != possessor_id
#                 and possessor_ball_dist < PASS_DISTANCE_THRESHOLD
#                 and (frame_index - ball_released_frame <= PASS_TIMEOUT)
#                 and player_team_map.get(ball_released_by, -1) == possessor_team  # Only same team
#                 and ball_travel_distance > MIN_PASS_TRAVEL_DIST                # Ball must travel enough
#                 and (ball_direction_angle is None or ball_direction_angle < MAX_PASS_ANGLE_CHANGE)  # Not a rebound
#             ):
#                 print(f"✅ PASS DETECTED: {ball_released_by} (team {ball_released_team}) -> {possessor_id} (team {possessor_team})")
#                 pass_counts[ball_released_by][possessor_id] += 1

#                 writer.writerow([
#                     frame_index,
#                     ball_released_by,
#                     ball_released_team,
#                     possessor_id,
#                     possessor_team,
#                     round(ball_velocity, 2),
#                     round(last_possessor_ball_dist, 2),
#                     round(possessor_ball_dist, 2)
#                 ])
#                 pass_armed = False  # reset

#             elif pass_armed and frame_index - ball_released_frame > PASS_TIMEOUT:
#                 print(f"Pass timeout: no receiver within {PASS_TIMEOUT} frames")
#                 pass_armed = False

#             # Debug for cases where no pass
#             if not pass_armed:
#                 print("No pass armed or detected")

#             # Until here, the above code is the one that replaces the previous -----------------------------

#             last_possessor_id = possessor_id
#             last_possessor_team = possessor_team
#             last_possessor_ball_dist = possessor_ball_dist

#             last_possessor_id = possessor_id
#             last_possessor_team = possessor_team
#             last_possessor_ball_dist = possessor_ball_dist

#             # Radar and yield
#             detections = sv.Detections.merge([players, referees])
#             color_lookup = np.concatenate([
#                 player_team_ids,
#                 np.full(len(referees), 2)
#             ])

#             radar_img = render_radar(detections, keypoints, color_lookup, colors=PASS_MAP_COLORS)
#             radar_img = add_legend(radar_img, COLORS[:2] + ['#FFD700'])

#             team_pass_total = {tid: 0 for tid in [0, 1]}
#             for from_pid, to_dict in pass_counts.items():
#                 from_team = player_team_map.get(from_pid, None)
#                 if from_team is not None:
#                     team_pass_total[from_team] += sum(to_dict.values())

#             yield frame, radar_img, team_pass_total

# def add_legend(radar_img, team_colors):
#     h, w, _ = radar_img.shape
#     y = 30
#     for tid, color in enumerate(team_colors):
#         rgb = tuple(int(color.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
#         if tid < 2:
#             label = f'Team {tid}'
#         else:
#             label = 'Referee'
#         cv2.circle(radar_img, (30, y), 10, rgb, -1)
#         cv2.putText(radar_img, label, (50, y + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
#         y += 30
#     return radar_img


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

# --- Tunable parameters for robust pass detection ---
BALL_HISTORY_LEN = 5
MIN_BALL_SPEED = 5  # pixels/frame
PASS_DISTANCE_THRESHOLD = 50
BALL_VELOCITY_THRESHOLD = 2  # pixels/frame
PASS_TIMEOUT = 20  # max frames after ball release
MIN_PASS_GAP = 3  # frames between release and reception
MIN_PASS_TRAVEL_DIST = 40  # pixels
MAX_PASS_ANGLE_CHANGE = 60 # degrees, to filter out rebounds

def run_pass_map(source_video_path: str, device: str):
    import csv

    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device)
    ball_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device)
    pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device)

    # Collect crops for team classification
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='Collecting crops for team classifier'):
        detections = sv.Detections.from_ultralytics(player_model(frame, imgsz=1280, verbose=False)[0])
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])

    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    # Setup for main video processing
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    ball_positions = deque(maxlen=BALL_HISTORY_LEN)
    player_positions = defaultdict(list)
    player_team_map = dict()
    pass_counts = defaultdict(lambda: defaultdict(int))

    last_possessor_id = None
    last_possessor_team = None
    last_possessor_ball_dist = None

    ball_released_by = None
    ball_released_team = None
    ball_released_frame = None
    pass_armed = False

    pass_log_path = "pass_log.csv"
    frame_index = 0

    with open(pass_log_path, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "frame_index",
            "from_player_id",
            "from_team",
            "to_player_id",
            "to_team",
            "ball_velocity",
            "from_dist",
            "to_dist"
        ])

        for frame in frame_generator:
            print(f"\n--- Frame {frame_index} ---")
            frame_index += 1

            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

            player_result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            tracked = tracker.update_with_detections(detections)

            players = tracked[tracked.class_id == PLAYER_CLASS_ID]
            referees = tracked[tracked.class_id == REFEREE_CLASS_ID]

            if len(players) == 0:
                print("No players detected")
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
                print("No ball detected")
                continue

            ball_centroid = balls.get_anchors_coordinates(anchor=sv.Position.CENTER)[0]
            ball_positions.append(ball_centroid)

            if len(ball_positions) < 2:
                print("Insufficient ball position history")
                continue

            # Compute ball velocity
            ball_velocity = np.linalg.norm(ball_positions[-1] - ball_positions[-2])
            print(f"Ball velocity: {ball_velocity:.2f}")

            # Find current closest player
            dists = np.linalg.norm(player_centroids - ball_centroid, axis=1)
            possessor_idx = np.argmin(dists)
            possessor_id = player_ids[possessor_idx]
            possessor_team = player_team_ids[possessor_idx]
            possessor_ball_dist = dists[possessor_idx]

            print(f"Closest player: ID {possessor_id}, Team {possessor_team}, Distance to ball: {possessor_ball_dist:.2f}")

            # --- Pass Logic ---

            # Calculate ball travel distance since release
            ball_travel_distance = 0
            if len(ball_positions) >= 2:
                ball_travel_distance = np.linalg.norm(ball_positions[-1] - ball_positions[0])

            # Calculate ball trajectory angle (if possible)
            ball_direction_angle = None
            if len(ball_positions) >= 3:
                vec1 = ball_positions[-2] - ball_positions[-3]
                vec2 = ball_positions[-1] - ball_positions[-2]
                if np.linalg.norm(vec1) > 0 and np.linalg.norm(vec2) > 0:
                    cos_angle = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
                    ball_direction_angle = np.arccos(np.clip(cos_angle, -1.0, 1.0)) * 180 / np.pi

            # Arm pass if ball is released with enough speed and possessor is close
            if (
                last_possessor_id is not None
                and ball_velocity > BALL_VELOCITY_THRESHOLD
                and last_possessor_ball_dist is not None
                and last_possessor_ball_dist < PASS_DISTANCE_THRESHOLD
                and np.linalg.norm(ball_positions[-1] - ball_positions[-2]) > BALL_VELOCITY_THRESHOLD
            ):
                print(f"Ball released by {last_possessor_id} (Team {last_possessor_team})")
                ball_released_by = last_possessor_id
                ball_released_team = last_possessor_team
                ball_released_frame = frame_index
                pass_armed = True

            # Detect pass with all improved conditions
            if (
                pass_armed
                and ball_released_by != possessor_id
                and possessor_ball_dist < PASS_DISTANCE_THRESHOLD
                and (frame_index - ball_released_frame >= MIN_PASS_GAP)
                and (frame_index - ball_released_frame <= PASS_TIMEOUT)
                and player_team_map.get(ball_released_by, -1) == possessor_team  # Only same team
                and possessor_team in [0, 1]  # Ignore referees/untracked
                and ball_travel_distance > MIN_PASS_TRAVEL_DIST  # Ball must travel enough
                and (ball_direction_angle is None or ball_direction_angle < MAX_PASS_ANGLE_CHANGE)  # Not a rebound
            ):
                print(f"✅ PASS DETECTED: {ball_released_by} (team {ball_released_team}) -> {possessor_id} (team {possessor_team})")
                pass_counts[ball_released_by][possessor_id] += 1

                writer.writerow([
                    frame_index,
                    ball_released_by,
                    ball_released_team,
                    possessor_id,
                    possessor_team,
                    round(ball_velocity, 2),
                    round(last_possessor_ball_dist, 2),
                    round(possessor_ball_dist, 2)
                ])
                pass_armed = False  # reset

            elif pass_armed and frame_index - ball_released_frame > PASS_TIMEOUT:
                print(f"Pass timeout: no receiver within {PASS_TIMEOUT} frames")
                pass_armed = False

            # Debug for cases where no pass
            if not pass_armed:
                print("No pass armed or detected")

            last_possessor_id = possessor_id
            last_possessor_team = possessor_team
            last_possessor_ball_dist = possessor_ball_dist

            # Radar and yield
            detections = sv.Detections.merge([players, referees])
            color_lookup = np.concatenate([
                player_team_ids,
                np.full(len(referees), 2)
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