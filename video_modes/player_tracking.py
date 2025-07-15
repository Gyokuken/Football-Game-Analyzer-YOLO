# import cv2
# import numpy as np
# import supervision as sv
# from ultralytics import YOLO

# PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
# ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
#     color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FF6347', '#FFD700']),
#     thickness=2
# )
# ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
#     color=sv.ColorPalette.from_hex(['#FF1493', '#00BFFF', '#FF6347', '#FFD700']),
#     text_color=sv.Color.from_hex('#FFFFFF'),
#     text_padding=5,
#     text_thickness=1,
#     text_position=sv.Position.BOTTOM_CENTER,
# )

# def run_player_tracking(source_video_path: str, device: str):
#     player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
#     frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
#     tracker = sv.ByteTrack(minimum_consecutive_frames=3)
#     for frame in frame_generator:
#         result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
#         detections = sv.Detections.from_ultralytics(result)
#         detections = tracker.update_with_detections(detections)
#         labels = [str(tracker_id) for tracker_id in detections.tracker_id]
#         annotated_frame = frame.copy()
#         annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
#         annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
#             annotated_frame, detections, labels=labels)
#         yield annotated_frame 


import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.common.team import TeamClassifier

PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'

# load your pre-trained classifier once
# classifier = TeamClassifier.load("team_classifier.pkl")

# map team_id → (name, BGR color)
TEAM_INFO = {
    0: ("Team White", (255, 255, 255)),
    1: ("Team Green", (  0, 128,   0)),
}

def run_player_tracking(source_video_path: str, device: str):
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)

    for frame in frame_generator:
        # 1️⃣ detect & track
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        # 2️⃣ crop each player ROI for classification
        crops = []
        for (x1, y1, x2, y2) in detections.xyxy:
            x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))
            crops.append(frame[y1:y2, x1:x2])

        # 3️⃣ predict team IDs in one batch
        team_ids = classifier.predict(crops)

        # 4️⃣ draw team-specific annotations
        annotated = frame.copy()
        for bbox, tid, team in zip(detections.xyxy, detections.tracker_id, team_ids):
            x1, y1, x2, y2 = map(int, bbox)
            name, color = TEAM_INFO[team]

            # draw ellipse at foot position
            foot_pt = ( (x1 + x2)//2, y2 )
            annotated = cv2.ellipse(
                annotated,
                center=foot_pt,
                axes=(15, 5),
                angle=0,
                startAngle=0,
                endAngle=360,
                color=color,
                thickness=2
            )

            # draw label with tracker ID
            annotated = cv2.putText(
                annotated,
                f"{name}#{tid}",
                (x1, y1 - 10),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.6,
                color=color,
                thickness=2
            )

        yield annotated
