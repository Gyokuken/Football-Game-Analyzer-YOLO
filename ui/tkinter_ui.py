import tkinter as tk
from tkinter import filedialog, ttk, messagebox
from threading import Thread, Event
from PIL import Image, ImageTk
import queue
import cv2
from video_modes.pitch_detection import run_pitch_detection
from video_modes.player_detection import run_player_detection
from video_modes.ball_detection import run_ball_detection
from video_modes.player_tracking import run_player_tracking
from video_modes.team_classification import run_team_classification, run_team_classification_with_possession
from video_modes.radar import run_radar
from video_modes.possession import PossessionTracker

import torch
import ultralytics.nn.tasks

TARGET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def custom_torch_safe_load(file):
    return torch.load(file, map_location=TARGET_DEVICE, weights_only=False), file

ultralytics.nn.tasks.torch_safe_load = custom_torch_safe_load

class Mode:
    PITCH_DETECTION = 'PITCH_DETECTION'
    PLAYER_DETECTION = 'PLAYER_DETECTION'
    BALL_DETECTION = 'BALL_DETECTION'
    PLAYER_TRACKING = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR = 'RADAR'


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
    mode_var = tk.StringVar(value=Mode.PLAYER_DETECTION)
    device_var = tk.StringVar(value='cpu')
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

    def process_video_thread(source_video_path, device, mode, frame_queue, stop_event, possession_tracker=None):
        try:
            if mode == Mode.PITCH_DETECTION:
                frame_generator = run_pitch_detection(source_video_path, device)
            elif mode == Mode.PLAYER_DETECTION:
                frame_generator = run_player_detection(source_video_path, device)
            elif mode == Mode.BALL_DETECTION:
                frame_generator = run_ball_detection(source_video_path, device)
            elif mode == Mode.PLAYER_TRACKING:
                frame_generator = run_player_tracking(source_video_path, device)
            elif mode == Mode.TEAM_CLASSIFICATION:
                for frame, possession in run_team_classification_with_possession(
                    source_video_path, device, stop_event, possession_tracker):
                    if stop_event.is_set():
                        break
                    frame_queue.put((frame, possession))
                return
            elif mode == Mode.RADAR:
                frame_generator = run_radar(source_video_path, device)
            else:
                raise NotImplementedError(f"Mode {mode} is not implemented.")
            for frame in frame_generator:
                if stop_event.is_set():
                    break
                frame_queue.put((frame, None))
        except Exception as e:
            frame_queue.put((e, None))

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
            device_var.get(),
            mode_var.get(),
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
    mode_menu = ttk.Combobox(controls_frame, textvariable=mode_var, values=[
        Mode.PITCH_DETECTION,
        Mode.PLAYER_DETECTION,
        Mode.BALL_DETECTION,
        Mode.PLAYER_TRACKING,
        Mode.TEAM_CLASSIFICATION,
        Mode.RADAR
    ], state='readonly')
    mode_menu.pack(side=tk.LEFT, padx=5)

    # Device selection
    tk.Label(controls_frame, text="Device:", bg='black', fg='white').pack(side=tk.LEFT, padx=5)
    device_menu = ttk.Combobox(controls_frame, textvariable=device_var, values=['cpu', 'cuda'], state='readonly')
    device_menu.pack(side=tk.LEFT, padx=5)

    # Media controls
    tk.Button(controls_frame, text="Play", command=start_processing).pack(side=tk.LEFT, padx=5)
    tk.Button(controls_frame, text="Pause", command=pause_processing).pack(side=tk.LEFT, padx=5)
    tk.Button(controls_frame, text="Resume", command=resume_processing).pack(side=tk.LEFT, padx=5)
    tk.Button(controls_frame, text="Stop", command=stop_processing).pack(side=tk.LEFT, padx=5)
    tk.Button(controls_frame, text="Remove Video", command=remove_video).pack(side=tk.LEFT, padx=5)

    root.mainloop() 