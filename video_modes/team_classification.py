import cv2
import numpy as np
import supervision as sv
from ultralytics import YOLO
from tqdm import tqdm
from sports.common.team import TeamClassifier
from sports.common.ball import BallTracker, BallAnnotator
from video_modes.possession import PossessionTracker

# Paths & IDs
PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
BALL_DETECTION_MODEL_PATH  = 'data/football-ball-detection.pt'
PLAYER_CLASS_ID     = 2
GOALKEEPER_CLASS_ID = 1
REFEREE_CLASS_ID    = 3
STRIDE              = 60


def get_crops(frame, detections):
    return [frame[int(y1):int(y2), int(x1):int(x2)]
            for x1, y1, x2, y2 in detections.xyxy]


def resolve_goalkeepers_team_id(players, team_ids, goalkeepers):
    pts = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    if len(pts) == 0:
        return np.zeros(len(goalkeepers), dtype=int)
    c0 = pts[team_ids == 0].mean(axis=0)
    c1 = pts[team_ids == 1].mean(axis=0)
    gk_pts = goalkeepers.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
    return np.array([0 if np.linalg.norm(p - c0) < np.linalg.norm(p - c1) else 1
                     for p in gk_pts], dtype=int)


def run_team_classification(source_video_path: str, device: str):
    # Load models
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device)
    ball_model   = YOLO(BALL_DETECTION_MODEL_PATH).to(device)
    tracker      = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=10, buffer_size=10)

    # 1️⃣ Gather player crops for classifier
    sampler = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    samples = []
    for frame in tqdm(sampler, desc='Sampling crops'):
        res = player_model(frame, imgsz=1280, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        players = dets[dets.class_id == PLAYER_CLASS_ID]
        samples += get_crops(frame, players)
        if len(samples) >= 500:
            break

    # 2️⃣ Fit TeamClassifier
    clf = TeamClassifier(device=device)
    clf.fit(samples)

    # 3️⃣ Main loop: detect, track, classify, annotate
    frames = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frames:
        # Player detection & tracking
        res = player_model(frame, imgsz=1280, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        dets = tracker.update_with_detections(dets)
        players = dets[dets.class_id == PLAYER_CLASS_ID]
        keepers = dets[dets.class_id == GOALKEEPER_CLASS_ID]
        referees= dets[dets.class_id == REFEREE_CLASS_ID]

        # Team classification
        crops    = get_crops(frame, players)
        p_ids    = clf.predict(crops)
        k_ids    = resolve_goalkeepers_team_id(players, p_ids, keepers)

        # Base annotation
        annotated = frame.copy()

        # Ball detection & annotation
        b_res = ball_model(frame, imgsz=640, verbose=False)[0]
        b_dets = sv.Detections.from_ultralytics(b_res)
        b_dets = ball_tracker.update(b_dets)
        annotated = ball_annotator.annotate(annotated, b_dets)

        # Draw players, keepers, referees
        all_dets   = sv.Detections.merge([players, keepers, referees])
        color_ids = np.concatenate([p_ids, k_ids, np.full(len(referees), REFEREE_CLASS_ID)], axis=0)
        for (x1,y1,x2,y2), tid, cid in zip(all_dets.xyxy, all_dets.tracker_id, color_ids):
            x1,y1,x2,y2 = map(int, (x1,y1,x2,y2))
            if cid == 0:   col=(255,255,255)
            elif cid == 1: col=(0,255,0)
            else:          col=(0,0,255)
            foot = ((x1+x2)//2, y2)
            cv2.ellipse(annotated, center=foot, axes=(15,5), angle=0,
                        startAngle=0, endAngle=360, color=col, thickness=2)
            cv2.putText(annotated, f"T{cid}#{tid}", (x1, y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)

        yield annotated


def run_team_classification_with_possession(
        source_video_path: str, device: str,
        stop_event, possession_tracker: PossessionTracker):
    if possession_tracker is None:
        raise ValueError("possession_tracker must not be None")
    # Load models
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device)
    ball_model   = YOLO(BALL_DETECTION_MODEL_PATH).to(device)
    tracker      = sv.ByteTrack(minimum_consecutive_frames=3)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=10, buffer_size=10)

    # Gather crops & fit classifier
    sampler = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    samples = []
    for frame in tqdm(sampler, desc='Sampling crops'):
        res = player_model(frame, imgsz=1280, verbose=False)[0]
        dets= sv.Detections.from_ultralytics(res)
        players = dets[dets.class_id == PLAYER_CLASS_ID]
        samples += get_crops(frame, players)
        if len(samples) >= 500: break
    clf = TeamClassifier(device=device)
    clf.fit(samples)

    # Main loop
    frames = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frames:
        if stop_event.is_set(): break
        # Detect & track
        res = player_model(frame, imgsz=1280, verbose=False)[0]
        dets = sv.Detections.from_ultralytics(res)
        dets = tracker.update_with_detections(dets)
        players = dets[dets.class_id == PLAYER_CLASS_ID]
        # Classify teams
        p_ids    = clf.predict(get_crops(frame, players))
        k_ids    = resolve_goalkeepers_team_id(players, p_ids, dets[dets.class_id == GOALKEEPER_CLASS_ID])
        # Ball detect & track
        b_res = ball_model(frame, imgsz=640, verbose=False)[0]
        b_dets= sv.Detections.from_ultralytics(b_res)
        b_dets= ball_tracker.update(b_dets)
        # Update possession
        if len(b_dets)>0 and len(players)>0:
            ball_xy   = b_dets.get_anchors_coordinates(sv.Position.CENTER)[0]
            p_xy      = players.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            idx       = np.argmin(np.linalg.norm(p_xy - ball_xy, axis=1))
            if idx < len(p_ids):
                possession_tracker.update(int(p_ids[idx]))
        # Annotate frame
        annotated = frame.copy()
        annotated = ball_annotator.annotate(annotated, b_dets)
        all_dets = sv.Detections.merge([players,
                                       dets[dets.class_id == GOALKEEPER_CLASS_ID],
                                       dets[dets.class_id == REFEREE_CLASS_ID]])
        color_ids = np.concatenate([p_ids, k_ids, np.full(len(all_dets)-len(p_ids)-len(k_ids), REFEREE_CLASS_ID)])
        for (x1,y1,x2,y2), tid, cid in zip(all_dets.xyxy,
                                           all_dets.tracker_id,
                                           color_ids):
            x1,y1,x2,y2 = map(int,(x1,y1,x2,y2))
            if cid==0:   col=(255,255,255)
            elif cid==1: col=(0,255,0)
            else:        col=(0,0,255)
            foot = ((x1+x2)//2, y2)
            cv2.ellipse(annotated, center=foot, axes=(15,5), angle=0,
                        startAngle=0, endAngle=360, color=col, thickness=2)
            cv2.putText(annotated, f"T{cid}#{tid}", (x1,y1-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        yield annotated, possession_tracker.get_percentages()
