import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from collections import defaultdict, deque
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.common.team import TeamClassifier

# Paths to models
PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
PITCH_DETECTION_MODEL_PATH = 'data/football-pitch-detection.pt'

# Pitch configuration
CONFIG = SoccerPitchConfiguration()
TEAM_COLORS = {0: (255, 255, 255), 1: (0, 255, 0)}  # Team 0 = white, Team 1 = green

# Speed estimation history
HISTORY_LEN = 5


def get_video_fps(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.release()
    return fps if fps > 0 else 30


def get_pitch_transformer(pitch_model, frame):
    pitch_result = pitch_model(frame, verbose=False)[0]
    keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    if np.sum(mask) < 4:
        return None
    return ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )


def get_crops(frame, detections):
    return [frame[int(y1):int(y2), int(x1):int(x2)] for x1, y1, x2, y2 in detections.xyxy]


def run_player_tracking(source_video_path: str, device: str):
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device)
    pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)

    fps = get_video_fps(source_video_path)
    print(f"[INFO] Auto-detected FPS: {fps:.2f}")

    position_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    prev_transformer = None
    frame_idx = 0

    # Collect team training crops
    print("[INFO] Collecting team classification crops...")
    crop_collector = sv.get_video_frames_generator(source_path=source_video_path, stride=60)
    crops = []
    for frame in crop_collector:
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        crops += get_crops(frame, detections)
        if len(crops) >= 500:
            break
    team_classifier = TeamClassifier(device=device)
    team_classifier.fit(crops)

    for frame in frame_generator:
        # Step 1: Get pitch transformer
        transformer = get_pitch_transformer(pitch_model, frame)
        if transformer is None:
            transformer = prev_transformer
        else:
            prev_transformer = transformer

        # Step 2: Detect & track
        result = player_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        if len(detections) == 0:
            yield frame
            frame_idx += 1
            continue

        # Step 3: Team classification
        crops = get_crops(frame, detections)
        team_ids = team_classifier.predict(crops)

        # Step 4: Speed estimation
        foot_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
        speeds = {}
        for i, tracker_id in enumerate(detections.tracker_id):
            if i >= len(foot_points):
                continue
            position_history[tracker_id].append((frame_idx, foot_points[i]))

            history = position_history[tracker_id]
            if len(history) >= 2 and transformer is not None:
                t0, p0 = history[0]
                t1, p1 = history[-1]
                dt = (t1 - t0) / fps
                if dt > 0:
                    m0 = transformer.transform_points(np.array([p0]))[0] / 100  # to meters
                    m1 = transformer.transform_points(np.array([p1]))[0] / 100
                    speeds[tracker_id] = np.linalg.norm(m1 - m0) / dt  # m/s

        # Step 5: Annotation
        annotated = frame.copy()
        for i, ((x1, y1, x2, y2), tracker_id, team_id) in enumerate(zip(detections.xyxy, detections.tracker_id, team_ids)):
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            cx, cy = (x1 + x2) // 2, y2
            foot = (cx, cy)

            color = TEAM_COLORS.get(team_id, (0, 0, 255))  # fallback to red
            cv2.ellipse(annotated, center=foot, axes=(15, 5), angle=0,
                        startAngle=0, endAngle=360, color=color, thickness=2)
                        
            if tracker_id in speeds:
                cv2.putText(annotated, f"{speeds[tracker_id]:.2f} m/s", (cx - 20, cy + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        frame_idx += 1
        yield annotated
