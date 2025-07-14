import argparse
from enum import Enum
from typing import Iterator, List

import torch
import ultralytics.nn.tasks

# Set your desired device (e.g., "cuda" or "cuda:0")
TARGET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_torch_safe_load(file):
    return torch.load(file, map_location=TARGET_DEVICE, weights_only=False), file

ultralytics.nn.tasks.torch_safe_load = custom_torch_safe_load


import os
import cv2
import numpy as np
import supervision as sv
from tqdm import tqdm
from ultralytics import YOLO

from sports.annotators.soccer import draw_pitch, draw_points_on_pitch
from sports.common.ball import BallTracker, BallAnnotator
from sports.common.team import TeamClassifier
from sports.common.view import ViewTransformer
from sports.configs.soccer import SoccerPitchConfiguration

import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from threading import Thread, Event
from PIL import Image, ImageTk
import queue

PARENT_DIR = os.path.dirname(os.path.abspath(__file__))
PLAYER_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-player-detection.pt')
PITCH_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-pitch-detection.pt')
BALL_DETECTION_MODEL_PATH = os.path.join(PARENT_DIR, 'data/football-ball-detection.pt')

BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

STRIDE = 60
CONFIG = SoccerPitchConfiguration()

COLORS = ['#FF1493', '#00BFFF', '#FF6347', '#FFD700']
VERTEX_LABEL_ANNOTATOR = sv.VertexLabelAnnotator(
    color=[sv.Color.from_hex(color) for color in CONFIG.colors],
    text_color=sv.Color.from_hex('#FFFFFF'),
    border_radius=5,
    text_thickness=1,
    text_scale=0.5,
    text_padding=5,
)
EDGE_ANNOTATOR = sv.EdgeAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    thickness=2,
    edges=CONFIG.edges,
)
TRIANGLE_ANNOTATOR = sv.TriangleAnnotator(
    color=sv.Color.from_hex('#FF1493'),
    base=20,
    height=15,
)
BOX_ANNOTATOR = sv.BoxAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
ELLIPSE_ANNOTATOR = sv.EllipseAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    thickness=2
)
BOX_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
)
ELLIPSE_LABEL_ANNOTATOR = sv.LabelAnnotator(
    color=sv.ColorPalette.from_hex(COLORS),
    text_color=sv.Color.from_hex('#FFFFFF'),
    text_padding=5,
    text_thickness=1,
    text_position=sv.Position.BOTTOM_CENTER,
)


class Mode(Enum):
    """
    Enum class representing different modes of operation for Soccer AI video analysis.
    """
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


def get_crops(frame: np.ndarray, detections: sv.Detections) -> List[np.ndarray]:
    """
    Extract crops from the frame based on detected bounding boxes.

    Args:
        frame (np.ndarray): The frame from which to extract crops.
        detections (sv.Detections): Detected objects with bounding boxes.

    Returns:
        List[np.ndarray]: List of cropped images.
    """
    return [sv.crop_image(frame, xyxy) for xyxy in detections.xyxy]


def resolve_goalkeepers_team_id(
    players: sv.Detections,
    players_team_id: np.array,
    goalkeepers: sv.Detections
) -> np.ndarray:
    """
    Resolve the team IDs for detected goalkeepers based on the proximity to team
    centroids.

    Args:
        players (sv.Detections): Detections of all players.
        players_team_id (np.array): Array containing team IDs of detected players.
        goalkeepers (sv.Detections): Detections of goalkeepers.

    Returns:
        np.ndarray: Array containing team IDs for the detected goalkeepers.

    This function calculates the centroids of the two teams based on the positions of
    the players. Then, it assigns each goalkeeper to the nearest team's centroid by
    calculating the distance between each goalkeeper and the centroids of the two teams.
    """
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


def render_radar(
    detections: sv.Detections,
    keypoints: sv.KeyPoints,
    color_lookup: np.ndarray
) -> np.ndarray:
    mask = (keypoints.xy[0][:, 0] > 1) & (keypoints.xy[0][:, 1] > 1)
    transformer = ViewTransformer(
        source=keypoints.xy[0][mask].astype(np.float32),
        target=np.array(CONFIG.vertices)[mask].astype(np.float32)
    )
    xy = detections.get_anchors_coordinates(anchor=sv.Position.BOTTOM_CENTER)
    transformed_xy = transformer.transform_points(points=xy)

    radar = draw_pitch(config=CONFIG)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 0],
        face_color=sv.Color.from_hex(COLORS[0]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 1],
        face_color=sv.Color.from_hex(COLORS[1]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 2],
        face_color=sv.Color.from_hex(COLORS[2]), radius=20, pitch=radar)
    radar = draw_points_on_pitch(
        config=CONFIG, xy=transformed_xy[color_lookup == 3],
        face_color=sv.Color.from_hex(COLORS[3]), radius=20, pitch=radar)
    return radar


def run_pitch_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run pitch detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    pitch_detection_model = YOLO(PITCH_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = pitch_detection_model(frame, verbose=False)[0]
        keypoints = sv.KeyPoints.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = VERTEX_LABEL_ANNOTATOR.annotate(
            annotated_frame, keypoints, CONFIG.labels)
        yield annotated_frame


def run_player_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player detection on a video and yield annotated frames.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)

        annotated_frame = frame.copy()
        annotated_frame = BOX_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = BOX_LABEL_ANNOTATOR.annotate(annotated_frame, detections)
        yield annotated_frame


def run_ball_detection(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run ball detection on a video and yield annotated frames.
    """
    ball_detection_model = YOLO(BALL_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    ball_tracker = BallTracker(buffer_size=20)
    ball_annotator = BallAnnotator(radius=6, buffer_size=10)

    def callback(image_slice: np.ndarray) -> sv.Detections:
        result = ball_detection_model(image_slice, imgsz=640, verbose=False)[0]
        return sv.Detections.from_ultralytics(result)

    # Remove overlap_filter_strategy if it causes errors, or set to a valid value
    slicer = sv.InferenceSlicer(
        callback=callback,
        slice_wh=(640, 640),
    )

    for frame in frame_generator:
        detections = slicer(frame).with_nms(threshold=0.1)
        detections = ball_tracker.update(detections)
        annotated_frame = frame.copy()
        annotated_frame = ball_annotator.annotate(annotated_frame, detections)
        yield annotated_frame


def run_player_tracking(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run player tracking on a video and yield annotated frames with tracked players.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
    player_detection_model = YOLO(PLAYER_DETECTION_MODEL_PATH).to(device=device)
    frame_generator = sv.get_video_frames_generator(source_path=source_video_path)
    tracker = sv.ByteTrack(minimum_consecutive_frames=3)
    for frame in frame_generator:
        result = player_detection_model(frame, imgsz=1280, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(result)
        detections = tracker.update_with_detections(detections)

        labels = [str(tracker_id) for tracker_id in detections.tracker_id]

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(annotated_frame, detections)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels=labels)
        yield annotated_frame


def run_team_classification(source_video_path: str, device: str) -> Iterator[np.ndarray]:
    """
    Run team classification on a video and yield annotated frames with team colors.

    Args:
        source_video_path (str): Path to the source video.
        device (str): Device to run the model on (e.g., 'cpu', 'cuda').

    Yields:
        Iterator[np.ndarray]: Iterator over annotated frames.
    """
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

        annotated_frame = frame.copy()
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels, custom_color_lookup=color_lookup)
        yield annotated_frame


def run_radar(source_video_path: str, device: str) -> Iterator[np.ndarray]:
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
        annotated_frame = ELLIPSE_ANNOTATOR.annotate(
            annotated_frame, detections, custom_color_lookup=color_lookup)
        annotated_frame = ELLIPSE_LABEL_ANNOTATOR.annotate(
            annotated_frame, detections, labels,
            custom_color_lookup=color_lookup)

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
        annotated_frame = sv.draw_image(annotated_frame, radar, opacity=0.5, rect=rect)
        yield annotated_frame


class PossessionTracker:
    def __init__(self):
        self.team_counts = [0, 0]
        self.total = 0
    def update(self, team_id):
        if team_id in [0, 1]:
            self.team_counts[team_id] += 1
            self.total += 1
    def get_percentages(self):
        if self.total == 0:
            return (0.0, 0.0)
        return (100 * self.team_counts[0] / self.total, 100 * self.team_counts[1] / self.total)


def process_video_thread(source_video_path, target_video_path, device, mode, frame_queue, stop_event, possession_tracker=None):
    try:
        if mode == Mode.PITCH_DETECTION:
            frame_generator = run_pitch_detection(
                source_video_path=source_video_path, device=device)
        elif mode == Mode.PLAYER_DETECTION:
            frame_generator = run_player_detection(
                source_video_path=source_video_path, device=device)
        elif mode == Mode.BALL_DETECTION:
            frame_generator = run_ball_detection(
                source_video_path=source_video_path, device=device)
        elif mode == Mode.PLAYER_TRACKING:
            frame_generator = run_player_tracking(
                source_video_path=source_video_path, device=device)
        elif mode == Mode.TEAM_CLASSIFICATION:
            # Use possession logic
            possession_tracker = PossessionTracker()
    
            for frame, possession in run_team_classification_with_possession(
                source_video_path, device, stop_event, possession_tracker):
                if stop_event.is_set():
                    break
                frame_queue.put((frame, possession))
            return
        elif mode == Mode.RADAR:
            frame_generator = run_radar(
                source_video_path=source_video_path, device=device)
        else:
            raise NotImplementedError(f"Mode {mode} is not implemented.")
        for frame in frame_generator:
            if stop_event.is_set():
                break
            frame_queue.put((frame, None))
    except Exception as e:
        frame_queue.put((e, None))


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


def start_tkinter_ui():
    root = tk.Tk()
    root.title("Sports Video Analyzer")
    root.geometry("1200x700")

    # Left frame for video
    left_frame = tk.Frame(root, width=800, height=700, bg='black')
    left_frame.pack(side=tk.LEFT, fill=tk.BOTH, expand=False)
    left_frame.pack_propagate(False)

    # Right frame for future features
    right_frame = tk.Frame(root, width=400, height=700, bg='gray90')
    right_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
    right_frame.pack_propagate(False)

    # Tabs on right
    notebook = ttk.Notebook(right_frame)
    notebook.pack(fill=tk.BOTH, expand=True)
    tab1 = tk.Frame(notebook)
    notebook.add(tab1, text="Team Possession %")
    possession_label = tk.Label(tab1, text="Team 0: 0.0%\nTeam 1: 0.0%", font=("Arial", 24), bg='gray90')
    possession_label.pack(pady=40)

    # Video display label
    video_label = tk.Label(left_frame, bg='black')
    video_label.pack(fill=tk.BOTH, expand=True)

    # Controls
    controls_frame = tk.Frame(left_frame, bg='black')
    controls_frame.pack(fill=tk.X, side=tk.BOTTOM)

    file_path_var = tk.StringVar()
    mode_var = tk.StringVar(value=Mode.PLAYER_DETECTION.value)
    device_var = tk.StringVar(value='cuda')
    stop_event = None
    frame_queue = queue.Queue(maxsize=2)
    thread = None
    paused = [False]  # Use a list for mutability in nested functions
    possession_tracker = [None]

    def select_file():
        file_path = filedialog.askopenfilename(filetypes=[('Video Files', '*.mp4 *.avi *.mov')])
        if file_path:
            file_path_var.set(file_path)
            clear_video()

    def start_processing():
        nonlocal thread, stop_event
        if not file_path_var.get():
            messagebox.showerror("Error", "Please select a video file.")
            return
        stop_processing()  # Stop any previous processing
        stop_event = Event()
        frame_queue.queue.clear()
        paused[0] = False
        if mode_var.get() == Mode.TEAM_CLASSIFICATION:
            possession_tracker[0] = PossessionTracker()
        else:
            possession_tracker[0] = None
        thread = Thread(target=process_video_thread, args=(
            file_path_var.get(),
            "output.mp4",  # Output path (not used in UI)
            device_var.get(),
            Mode(mode_var.get()),
            frame_queue,
            stop_event,
            possession_tracker[0]
        ), daemon=True)
        thread.start()
        update_video()

    def pause_processing():
        paused[0] = True

    def resume_processing():
        if paused[0]:
            paused[0] = False
            update_video()

    def stop_processing():
        nonlocal stop_event
        if stop_event is not None:
            stop_event.set()
        paused[0] = False

    def remove_video():
        stop_processing()
        file_path_var.set("")
        clear_video()

    def clear_video():
        video_label.config(image="")
        video_label.imgtk = None
        possession_label.config(text="Team 0: 0.0%\nTeam 1: 0.0%")

    def update_video():
        if paused[0]:
            return
        try:
            frame, possession = frame_queue.get_nowait()
            if isinstance(frame, Exception):
                messagebox.showerror("Error", str(frame))
                return
            left_frame.update_idletasks()
            display_w = left_frame.winfo_width()
            display_h = left_frame.winfo_height()
            h, w, _ = frame.shape
            scale = min(display_w / w, display_h / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            imgtk = ImageTk.PhotoImage(image=img)
            video_label.imgtk = imgtk
            video_label.config(image=imgtk)
            # Only update possession label in TEAM_CLASSIFICATION mode and if possession is not None
            if mode_var.get() == Mode.TEAM_CLASSIFICATION and possession is not None:
                p0, p1 = possession
                possession_label.config(text=f"Team 0: {p0:.1f}%\nTeam 1: {p1:.1f}%")
            else:
                possession_label.config(text="Team 0: 0.0%\nTeam 1: 0.0%")
        except queue.Empty:
            pass
        if thread and thread.is_alive() and not paused[0]:
            root.after(30, update_video)

    # File selection
    tk.Button(controls_frame, text="Select Video", command=select_file).pack(side=tk.LEFT, padx=5, pady=5)
    tk.Label(controls_frame, textvariable=file_path_var, bg='black', fg='white').pack(side=tk.LEFT, padx=5)

    # Mode selection
    tk.Label(controls_frame, text="Mode:", bg='black', fg='white').pack(side=tk.LEFT, padx=5)
    mode_menu = ttk.Combobox(controls_frame, textvariable=mode_var, values=[m.value for m in Mode], state='readonly')
    mode_menu.pack(side=tk.LEFT, padx=5)

    # Device selection
    tk.Label(controls_frame, text="Device:", bg='black', fg='white').pack(side=tk.LEFT, padx=5)
    device_menu = ttk.Combobox(controls_frame, textvariable=device_var, values=['cuda'], state='readonly')
    device_menu.pack(side=tk.LEFT, padx=5)

    # Media controls
    tk.Button(controls_frame, text="Play", command=start_processing).pack(side=tk.LEFT, padx=5)
    tk.Button(controls_frame, text="Pause", command=pause_processing).pack(side=tk.LEFT, padx=5)
    tk.Button(controls_frame, text="Resume", command=resume_processing).pack(side=tk.LEFT, padx=5)
    tk.Button(controls_frame, text="Stop", command=stop_processing).pack(side=tk.LEFT, padx=5)
    tk.Button(controls_frame, text="Remove Video", command=remove_video).pack(side=tk.LEFT, padx=5)

    root.mainloop()

# Only run the UI if this script is executed directly
if __name__ == '__main__':
    start_tkinter_ui()
