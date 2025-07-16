import numpy as np
import cv2
from ultralytics import YOLO
from collections import defaultdict, deque
from sports.configs.soccer import SoccerPitchConfiguration
from sports.common.view import ViewTransformer
from sports.common.team import TeamClassifier
import supervision as sv

PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
PITCH_DETECTION_MODEL_PATH = 'data/football-pitch-detection.pt'
CONFIG = SoccerPitchConfiguration()
TEAM_COLORS = {0: (255, 255, 255), 1: (0, 255, 0)}
HISTORY_LEN = 5

TRACK_COLOR = sv.Color.YELLOW
PLAYER_COLOR = sv.Color.RED


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


def individual_player_tracking(source_video_path: str, device: str, selected_id: int):
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device)
    pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    fps = get_video_fps(source_video_path)

    position_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    screen_history = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    prev_transformer = None
    frame_idx = 0
    total_distance = defaultdict(float)
    last_position = {}

    for frame in frame_generator:
        try:
            transformer = get_pitch_transformer(pitch_model, frame)
            if transformer is None:
                transformer = prev_transformer
            else:
                prev_transformer = transformer

            result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(result)
            detections = tracker.update_with_detections(detections)

            annotated = frame.copy()
            found = False

            foot_points = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)

            # First, show all player IDs in white
            for i, tracker_id in enumerate(detections.tracker_id):
                x1, y1, x2, y2 = map(int, detections.xyxy[i])
                # Show ID at top of each player with white background
                cv2.rectangle(annotated, (x1-5, y1-30), (x1+50, y1-5), (255, 255, 255), -1)  # White background
                cv2.putText(annotated, f"ID: {tracker_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)  # Black text on white background

            # Then highlight the selected player
            for i, tracker_id in enumerate(detections.tracker_id):
                if int(tracker_id) != selected_id:
                    continue

                found = True
                foot_point = tuple(map(int, foot_points[i]))
                x1, y1, x2, y2 = map(int, detections.xyxy[i])

                # Highlight selected player with red ellipse (no rectangle)
                cv2.ellipse(annotated, center=foot_point, axes=(15, 6), angle=0,
                            startAngle=0, endAngle=360, color=(0, 0, 255), thickness=3)
                
                # Overwrite the ID with red background for selected player
                cv2.rectangle(annotated, (x1-5, y1-30), (x1+50, y1-5), (0, 0, 255), -1)  # Red background
                cv2.putText(annotated, f"ID: {selected_id}", (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)  # White text on red background

                screen_history[tracker_id].append(foot_point)

                if transformer is not None:
                    foot_real = transformer.transform_points(np.array([foot_point], dtype=np.float32))[0] / 100.0
                    position_history[tracker_id].append((frame_idx, foot_real))

                    if len(position_history[tracker_id]) >= 2:
                        t0, p0 = position_history[tracker_id][0]
                        t1, p1 = position_history[tracker_id][-1]
                        dt = (t1 - t0) / fps
                        speed = np.linalg.norm(p1 - p0) / dt if dt > 0 else 0.0

                        cv2.putText(
                            annotated,
                            f"{speed:.2f} m/s",
                            (foot_point[0], foot_point[1] + 25),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.7,
                            (255, 255, 255),
                            2
                        )

                        if tracker_id in last_position:
                            distance_step = np.linalg.norm(p1 - last_position[tracker_id])
                            total_distance[tracker_id] += distance_step
                        last_position[tracker_id] = p1

                        cv2.putText(
                            annotated,
                            f"Distance: {total_distance[tracker_id]:.1f}m",
                            (annotated.shape[1] - 200, 30),  # Top-right position
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (255, 255, 255),
                            2
                        )

                    if len(screen_history[tracker_id]) >= 2:
                        for j in range(1, len(screen_history[tracker_id])):
                            cv2.line(
                                annotated,
                                screen_history[tracker_id][j - 1],
                                screen_history[tracker_id][j],
                                (0, 255, 255),  # Yellow track
                                2
                            )

            if not found:
                cv2.putText(
                    annotated,
                    f"Player ID {selected_id} not detected",
                    (30, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 0, 255),
                    2
                )

            frame_idx += 1
            yield frame, annotated

        except Exception as e:
            print(f"[ERROR] Frame {frame_idx}: {e}")
            yield frame, frame
            frame_idx += 1