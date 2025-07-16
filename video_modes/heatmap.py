import numpy as np
import supervision as sv
from ultralytics import YOLO
from sports.configs.soccer import SoccerPitchConfiguration
from sports.annotators.soccer import draw_pitch
from sports.common.view import ViewTransformer
import cv2

PLAYER_DETECTION_MODEL_PATH = 'data/football-player-detection.pt'
PITCH_DETECTION_MODEL_PATH = 'data/football-pitch-detection.pt'
BALL_DETECTION_MODEL_PATH = 'data/football-ball-detection.pt'
PLAYER_CLASS_ID = 2
BALL_CLASS_ID = 0
STRIDE = 60
CONFIG = SoccerPitchConfiguration()

def run_heatmap(source_video_path: str, device: str):
    player_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device)
    pitch_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device)
    ball_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device)

    frame_generator = sv.get_video_frames_generator(source_path=source_video_path, stride=STRIDE)
    
    team0_positions = []
    team1_positions = []

    for frame in frame_generator:
        try:
            # --- Detect pitch ---
            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)
            mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
            if mask.sum() < 4:
                continue
            transformer = ViewTransformer(
                source=keypoints.xy[0][mask].astype(np.float32),
                target=np.array(CONFIG.vertices)[mask].astype(np.float32)
            )

            # --- Detect ball ---
            ball_result = ball_model(frame, imgsz=1280, verbose=False)[0]
            ball_dets = sv.Detections.from_ultralytics(ball_result)
            ball_candidates = ball_dets[ball_dets.class_id == BALL_CLASS_ID]
            if len(ball_candidates) == 0:
                continue
            ball_centers = ball_candidates.get_anchors_coordinates(anchor=sv.Position.CENTER)
            ball_on_pitch = transformer.transform_points(points=ball_centers)
            if not is_ball_in_pitch(ball_on_pitch[0], CONFIG):
                continue

            # --- Detect players ---
            player_result = player_model(frame, imgsz=1280, verbose=False)[0]
            detections = sv.Detections.from_ultralytics(player_result)
            players = detections[detections.class_id == PLAYER_CLASS_ID]
            if len(players) == 0:
                continue

            player_centroids = players.get_anchors_coordinates(anchor=sv.Position.CENTER)
            player_centroids_on_pitch = transformer.transform_points(points=player_centroids)

            # --- Team classification ---
            team_ids = classify_teams(players, frame)
            for i, team_id in enumerate(team_ids):
                if team_id == 0:
                    team0_positions.append(player_centroids_on_pitch[i:i+1])
                elif team_id == 1:
                    team1_positions.append(player_centroids_on_pitch[i:i+1])

            overlay0, overlay1 = render_dual_heatmap(team0_positions, team1_positions, CONFIG)
            yield frame, overlay0, overlay1

        except Exception as e:
            print(f"[Heatmap Error] {e}")
            continue


def is_ball_in_pitch(ball_point, config):
    x, y = ball_point
    return 0 <= x <= config.length and 0 <= y <= config.width


def classify_teams(players: sv.Detections, frame: np.ndarray):
    # Use team classifier logic here; placeholder assigns dummy teams
    num_players = len(players)
    return np.array([i % 2 for i in range(num_players)])  # Mock: even-index -> team 0, odd-index -> team 1


def render_dual_heatmap(team0_positions, team1_positions, config):
    pitch = draw_pitch(config=config)

    def create_heatmap(points, color_map):
        if not points:
            return np.zeros_like(pitch)
        all_points = np.vstack(points)
        x_bins = np.linspace(0, config.length, 150)
        y_bins = np.linspace(0, config.width, 75)
        heatmap, _, _ = np.histogram2d(all_points[:, 0], all_points[:, 1], bins=[x_bins, y_bins])
        heatmap = cv2.GaussianBlur(heatmap.T, (0, 0), sigmaX=3, sigmaY=3)
        normalized = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        colored = cv2.applyColorMap(normalized, color_map)
        return colored

    heatmap0 = create_heatmap(team0_positions, cv2.COLORMAP_INFERNO)  # Red/yellow
    heatmap1 = create_heatmap(team1_positions, cv2.COLORMAP_TURBO)    # Blue/green

    # Resize heatmaps to match pitch size if needed
    if heatmap0.shape[:2] != pitch.shape[:2]:
        heatmap0 = cv2.resize(heatmap0, (pitch.shape[1], pitch.shape[0]))
    if heatmap1.shape[:2] != pitch.shape[:2]:
        heatmap1 = cv2.resize(heatmap1, (pitch.shape[1], pitch.shape[0]))

    overlay0 = cv2.addWeighted(pitch, 0.7, heatmap0, 0.3, 0)
    overlay1 = cv2.addWeighted(pitch, 0.7, heatmap1, 0.3, 0)

    return overlay0, overlay1
