import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from sports.common.team import TeamClassifier
from video_modes.team_classification import get_crops, resolve_goalkeepers_team_id
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch, draw_points_on_pitch

PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
PITCH_DETECTION_MODEL_PATH = 'data/football-pitch-detection.pt'
PLAYER_CLASS_ID = 2
GOALKEEPER_CLASS_ID = 1
REFEREE_CLASS_ID = 3
STRIDE = 60
CONFIG = SoccerPitchConfiguration()
COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']


def render_radar(detections, keypoints, color_lookup, colors=COLORS):
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    from sports.common.view import ViewTransformer
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)
    radar = draw_pitch(config=CONFIG)
    for i, color in enumerate(colors):
        radar = draw_points_on_pitch(
            config=CONFIG, xy=transformed_xy[color_lookup == i],
            face_color=sv.Color.from_hex(color), radius=20, pitch=radar)
    return radar

def run_radar(source_video_path: str, device: str):
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
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
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)
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
        annotated_frame = frame.copy()
        # (Annotation code omitted for brevity, add as needed)
        h, w, _ = frame.shape
        radar = render_radar(detections, keypoints, color_lookup)
        radar = sv.resize_image(radar, (w // 2, h // 2))
        radar_h, radar_w, _ = radar.shape
        rect = sv.Rect(
            x=w // 2 - radar_w // 2,
            y=h - radar_h,
            width=radar_w,
            height=radar_h
        )
        #annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame , radar