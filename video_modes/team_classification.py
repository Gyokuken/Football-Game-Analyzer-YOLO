import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from sports.common.team import TeamClassifier
from sports.common.ball import BallTracker
from video_modes.possession import PossessionTracker

PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
BALL_DETECTION_MODEL_PATH = 'data/football-ball-detection.pt'
PLAYER_CLASS_ID = 2
GOALKEEPER_CLASS_ID = 1
REFEREE_CLASS_ID = 3
STRIDE = 60

def get_crops(frame, detections):
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]

def resolve_goalkeepers_team_id(players, players_team_id, goalkeepers):
    goalkeepers_xy = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    team_0_centroid = players_xy[players_team_id == 0].mean(axis=0)
    team_1_centroid = players_xy[players_team_id == 1].mean(axis=0)
    goalkeepers_team_id = []
    for goalkeeper_xy in goalkeepers_xy:
        dist_0 = np.linalg.norm(goalkeeper_xy - team_0_centroid)
        dist_1 = np.linalg.norm(goalkeeper_xy - team_1_centroid)
        goalkeepers_team_id.append(0 if dist_0 < dist_1 else 1)
    return np.array(goalkeepers_team_id)

def run_team_classification(source_video_path: str, device: str):
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(
        source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)
        goalkeepers = detections[detections.class_id == GOALKEEPER_CLASS_ID]
        goalkeepers_team_id = resolve_goalkeepers_team_id(
            players, players_team_id, goalkeepers)
        referees = detections[detections.class_id == REFEREE_CLASS_ID]
        detections = sv.Detections.merge([players, goalkeepers, referees])
        color_lookup = np.array(
                players_team_id.tolist() +
                goalkeepers_team_id.tolist() +
                [REFEREE_CLASS_ID] * len(referees)
        )
        labels = [str(tracker_id) for tracker_id in detections.tracker_id]
        # (Annotation code omitted for brevity, add as needed)
        yield frame

def run_team_classification_with_possession(source_video_path, device, stop_event, possession_tracker):
    if possession_tracker is None:
        raise ValueError("possession_tracker must not be None in TEAM_CLASSIFICATION mode")
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    crops = []
    for frame in tqdm(frame_generator, desc='collecting crops'):
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections[detections.class_id == PLAYER_CLASS_ID])
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=10)
    for frame in frame_generator:
        if stop_event.is_set():
            break
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)
        players = detections[detections.class_id == PLAYER_CLASS_ID]
        crops = get_crops(frame, players)
        players_team_id = team_classifier.predict(crops)
        # Ball detection
        ball_result = ball_detection_model(frame, imgsz=640, verbose=False)[0]
        ball_detections = sv.Detections.from_ultralytics(ball_result)
        ball_detections = ball_tracker.update(ball_detections)
        # Possession logic
        if len(ball_detections) > 0 and len(players) > 0:
            ball_xy = ball_detections.get_anchors_coordinates(sv.Position.CENTER)[0]
            players_xy = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            dists = np.linalg.norm(players_xy - ball_xy, axis=1)
            closest_idx = np.argmin(dists)
            if closest_idx < len(players_team_id):
                team_id = players_team_id[closest_idx]
                possession_tracker.update(team_id)
        yield frame, possession_tracker.get_percentages() 