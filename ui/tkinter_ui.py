# import tkinter as tk
# from tkinter import filedialog, ttk, messagebox
# from threading import Thread, Event
# from PIL import Image, ImageTk
# import queue
# import cv2
# import os
# import numpy as np

# import torch
# import ultralytics.nn.tasks

# from video_modes.pitch_detection import run_pitch_detection
# from video_modes.player_detection import run_player_detection
# from video_modes.ball_detection import run_ball_detection
# from video_modes.player_tracking import run_player_tracking
# from video_modes.team_classification import run_team_classification, run_team_classification_with_possession
# from video_modes.radar import run_radar
# from video_modes.pass_map import run_pass_map
# from video_modes.possession import PossessionTracker

# # Ensure models load to the correct device
# TARGET_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Override Ultralytics safe load to respect our device
# def custom_torch_safe_load(file):
#     return torch.load(file, map_location=TARGET_DEVICE, weights_only=False), file
# ultralytics.nn.tasks.torch_safe_load = custom_torch_safe_load

# class Mode:
#     PITCH_DETECTION     = 'PITCH_DETECTION'
#     PLAYER_DETECTION    = 'PLAYER_DETECTION'
#     BALL_DETECTION      = 'BALL_DETECTION'
#     PLAYER_TRACKING     = 'PLAYER_TRACKING'
#     TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
#     RADAR               = 'RADAR'
#     PASS_MAP            = 'PASS_MAP'
#     HEATMAP             = 'HEATMAP'


# def start_tkinter_ui():
#     root = tk.Tk()
#     root.title("Sports Video Analyzer")
#     root.geometry("1200x700")

#     # Left frame for video display
#     left_frame = tk.Frame(root, width=800, height=700, bg='black')
#     left_frame.pack(side=tk.LEFT, fill=tk.BOTH)
#     left_frame.pack_propagate(False)
#     video_label = tk.Label(left_frame, bg='black')
#     video_label.pack(fill=tk.BOTH, expand=True)

#     # Right frame with tabs
#     right_frame = tk.Frame(root, width=400, height=700, bg='gray90')
#     right_frame.pack(side=tk.RIGHT, fill=tk.BOTH)
#     notebook = ttk.Notebook(right_frame)
#     notebook.pack(fill=tk.BOTH, expand=True)

#     # Possession tab
#     tab_pos = tk.Frame(notebook, bg='gray90')
#     notebook.add(tab_pos, text="Team Possession %")
#     possession_label = tk.Label(tab_pos, text="Team 0: 0.0%\nTeam 1: 0.0%",
#                                 font=("Arial",24), bg='gray90')
#     possession_label.pack(pady=40)

#     # Radar tab
#     tab_radar = tk.Frame(notebook, bg='gray90')
#     notebook.add(tab_radar, text="Radar")
#     radar_label = tk.Label(tab_radar, bg='gray90')
#     radar_label.pack(pady=20)

#     # Pass Map tab
#     tab_pass = tk.Frame(notebook, bg='gray90')
#     notebook.add(tab_pass, text="Pass Map")
#     pass_map_label = tk.Label(tab_pass, bg='gray90')
#     pass_map_label.pack(pady=20)
#     pass_counts_label = tk.Label(tab_pass, text="", font=("Arial",16), bg='gray90')
#     pass_counts_label.pack(pady=5)

#     # Heatmap tab
# # In Heatmap tab creation (around line 110)
# # --- UPDATED ---
#     tab_heatmap = tk.Frame(notebook, bg='gray90')
#     notebook.add(tab_heatmap, text="Heatmap")
#     heatmap_label_team0 = tk.Label(tab_heatmap, bg='gray90')
#     heatmap_label_team0.pack(pady=10)
#     heatmap_label_team1 = tk.Label(tab_heatmap, bg='gray90')
#     heatmap_label_team1.pack(pady=10)

#     # Controls frame
#     controls = tk.Frame(left_frame, bg='black')
#     controls.pack(fill=tk.X, side=tk.BOTTOM)

#     # State variables
#     file_path_var = tk.StringVar()
#     file_name_var = tk.StringVar()
#     mode_var      = tk.StringVar(value=Mode.PLAYER_DETECTION)
#     device_var    = tk.StringVar(value='cpu')
#     stop_event    = None
#     frame_queue   = queue.Queue(maxsize=2)
#     thread        = None
#     paused        = [False]
#     possession_tracker = None

#     # Helper: clear all displays
# # In clear_video() function
#     # --- UPDATED ---
#     def clear_video():
#         video_label.config(image=None)
#         video_label.imgtk = None
#         possession_label.config(text="Team 0: 0.0%\nTeam 1: 0.0%")
#         radar_label.config(image=None)
#         radar_label.imgtk = None
#         pass_map_label.config(image=None)
#         pass_map_label.imgtk = None
#         pass_counts_label.config(text="")
#         heatmap_label_team0.config(image=None)
#         heatmap_label_team0.imgtk = None
#         heatmap_label_team1.config(image=None)
#         heatmap_label_team1.imgtk = None


#     # Select video file
#     def select_file():
#         path = filedialog.askopenfilename(filetypes=[('Video Files','*.mp4 *.avi *.mov')])
#         if path:
#             file_path_var.set(path)
#             file_name_var.set(os.path.basename(path))
#             file_label.config(text=file_name_var.get())
#             clear_video()

#     # Background processing thread
#     def process_video_thread(src, dev, mode, q, stop_evt, poss_tracker=None):
#         try:
#             if mode == Mode.PITCH_DETECTION:
#                 gen = run_pitch_detection(src, dev)
#             elif mode == Mode.PLAYER_DETECTION:
#                 gen = run_player_detection(src, dev)
#             elif mode == Mode.BALL_DETECTION:
#                 gen = run_ball_detection(src, dev)
#             elif mode == Mode.PLAYER_TRACKING:
#                 gen = run_player_tracking(src, dev)
#             elif mode == Mode.TEAM_CLASSIFICATION:
#                 gen = run_team_classification_with_possession(src, dev, stop_evt, poss_tracker)
#             elif mode == Mode.RADAR:
#                 gen = run_radar(src, dev)
#             elif mode == Mode.PASS_MAP:
#                 gen = run_pass_map(src, dev)
#             elif mode == Mode.HEATMAP:
#                 from video_modes.heatmap import run_heatmap
#                 gen = run_heatmap(src, dev)
#             else:
#                 raise NotImplementedError(f"Mode {mode} not implemented")
#             for out in gen:
#                 if stop_evt.is_set():
#                     break
#                 frame_queue.put(out)
#         except Exception as e:
#             frame_queue.put((e, None))

#     # Start video processing
#     def start_processing():
#         nonlocal thread, stop_event, possession_tracker
#         if not file_path_var.get():
#             messagebox.showerror("Error","Select a video file.")
#             return
#         # Stop previous
#         if stop_event:
#             stop_event.set()
#         stop_event = Event()
#         frame_queue.queue.clear()
#         paused[0] = False
#         # Possession tracker only for team classification
#         if mode_var.get() == Mode.TEAM_CLASSIFICATION:
#             possession_tracker = PossessionTracker()
#         else:
#             possession_tracker = None
#         thread = Thread(target=process_video_thread,
#                         args=(file_path_var.get(), device_var.get(), mode_var.get(),
#                               frame_queue, stop_event, possession_tracker), daemon=True)
#         thread.start()
#         update_video()

#     def pause_processing():
#         paused[0] = True
#     def resume_processing():
#         if paused[0]:
#             paused[0] = False
#             update_video()
#     def stop_processing():
#         if stop_event:
#             stop_event.set()
#         paused[0] = False

#     # Display loop
#     def update_video():
#         if paused[0]: return
#         try:
#             out = frame_queue.get_nowait()
#         except queue.Empty:
#             pass
#         else:
#             mode = mode_var.get()
#             # --- Pass Map Mode ---
#             if mode == Mode.PASS_MAP:
#                 # Expect out to be (frame, pass_map_img, team_pass_counts)
#                 if isinstance(out, tuple) and len(out) == 3:
#                     video_frame, pass_map_img, team_pass_counts = out
#                 else:
#                     # Fallbacks for legacy or error cases
#                     video_frame = out if isinstance(out, np.ndarray) else np.zeros((480, 640, 3), dtype=np.uint8)
#                     pass_map_img = np.zeros_like(video_frame)
#                     team_pass_counts = {0: 0, 1: 0}
#                 # Display video frame on the left
#                 left_frame.update_idletasks()
#                 display_w = left_frame.winfo_width()
#                 display_h = left_frame.winfo_height()
#                 h, w, _ = video_frame.shape
#                 scale = min(display_w / w, display_h / h)
#                 new_w, new_h = int(w * scale), int(h * scale)
#                 img = cv2.cvtColor(video_frame, cv2.COLOR_BGR2RGB)
#                 img = Image.fromarray(img)
#                 img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
#                 imgtk = ImageTk.PhotoImage(image=img)
#                 setattr(video_label, 'imgtk', imgtk)
#                 video_label.config(image=imgtk)
#                 # Display pass map in pass map tab
#                 pm_h, pm_w, _ = pass_map_img.shape
#                 pm_scale = min(350 / pm_w, 350 / pm_h, 1.0)
#                 pm_new_w, pm_new_h = int(pm_w * pm_scale), int(pm_h * pm_scale)
#                 pm_img = cv2.cvtColor(pass_map_img, cv2.COLOR_BGR2RGB)
#                 pm_pil = Image.fromarray(pm_img)
#                 pm_pil = pm_pil.resize((pm_new_w, pm_new_h), Image.Resampling.LANCZOS)
#                 pm_imgtk = ImageTk.PhotoImage(image=pm_pil)
#                 setattr(pass_map_label, 'imgtk', pm_imgtk)
#                 pass_map_label.config(image=pm_imgtk)
#                 # Display team pass counts
#                 pass_counts_label.config(text=f"Team 0 passes: {team_pass_counts.get(0,0)}   Team 1 passes: {team_pass_counts.get(1,0)}")
#                 possession_label.config(text="Team 0: 0.0%\nTeam 1: 0.0%")
#             # --- Radar Mode ---
#             elif mode == Mode.RADAR:
#                 if isinstance(out, tuple) and len(out) == 2:
#                     annotated_frame, radar_img = out
#                     left_frame.update_idletasks()
#                     display_w = left_frame.winfo_width()
#                     display_h = left_frame.winfo_height()
#                     h, w, _ = annotated_frame.shape
#                     scale = min(display_w / w, display_h / h)
#                     new_w, new_h = int(w * scale), int(h * scale)
#                     img = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
#                     img = Image.fromarray(img)
#                     img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
#                     imgtk = ImageTk.PhotoImage(image=img)
#                     setattr(video_label, 'imgtk', imgtk)
#                     video_label.config(image=imgtk)
#                     # Display radar in radar tab
#                     radar_h, radar_w, _ = radar_img.shape
#                     radar_scale = min(350 / radar_w, 350 / radar_h, 1.0)
#                     radar_new_w, radar_new_h = int(radar_w * radar_scale), int(radar_h * radar_scale)
#                     radar_img_rgb = cv2.cvtColor(radar_img, cv2.COLOR_BGR2RGB)
#                     radar_pil = Image.fromarray(radar_img_rgb)
#                     radar_pil = radar_pil.resize((radar_new_w, radar_new_h), Image.Resampling.LANCZOS)
#                     radar_imgtk = ImageTk.PhotoImage(image=radar_pil)
#                     setattr(radar_label, 'imgtk', radar_imgtk)
#                     radar_label.config(image=radar_imgtk)
#                     possession_label.config(text="Team 0: 0.0%\nTeam 1: 0.0%")
#             # --- Team Classification Mode ---
#             elif mode == Mode.TEAM_CLASSIFICATION:
#                 if isinstance(out, tuple) and len(out) == 2:
#                     frame, possession = out
#                     left_frame.update_idletasks()
#                     display_w = left_frame.winfo_width()
#                     display_h = left_frame.winfo_height()
#                     h, w, _ = frame.shape
#                     scale = min(display_w / w, display_h / h)
#                     new_w, new_h = int(w * scale), int(h * scale)
#                     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     img = Image.fromarray(img)
#                     img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
#                     imgtk = ImageTk.PhotoImage(image=img)
#                     setattr(video_label, 'imgtk', imgtk)
#                     video_label.config(image=imgtk)
#                     p0, p1 = possession
#                     possession_label.config(text=f"Team 0: {p0:.1f}%\nTeam 1: {p1:.1f}%")

#             elif mode == Mode.HEATMAP:
#                 if isinstance(out, tuple) and len(out) == 3:
#                     frame, heatmap0, heatmap1 = out
#                     left_frame.update_idletasks()
#                     display_w = left_frame.winfo_width()
#                     display_h = left_frame.winfo_height()
#                     h, w, _ = frame.shape
#                     scale = min(display_w / w, display_h / h)
#                     new_w, new_h = int(w * scale), int(h * scale)
#                     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                     img = Image.fromarray(img)
#                     img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
#                     imgtk = ImageTk.PhotoImage(image=img)
#                     setattr(video_label, 'imgtk', imgtk)
#                     video_label.config(image=imgtk)

#                     # Heatmap 0 (Team 0)
#                     h0, w0, _ = heatmap0.shape
#                     scale0 = min(350 / w0, 350 / h0, 1.0)
#                     new_w0, new_h0 = int(w0 * scale0), int(h0 * scale0)
#                     img0 = cv2.cvtColor(heatmap0, cv2.COLOR_BGR2RGB)
#                     img0 = Image.fromarray(img0)
#                     img0 = img0.resize((new_w0, new_h0), Image.Resampling.LANCZOS)
#                     imgtk0 = ImageTk.PhotoImage(image=img0)
#                     setattr(heatmap_label_team0, 'imgtk', imgtk0)
#                     heatmap_label_team0.config(image=imgtk0)

#                     # Heatmap 1 (Team 1)
#                     h1, w1, _ = heatmap1.shape
#                     scale1 = min(350 / w1, 350 / h1, 1.0)
#                     new_w1, new_h1 = int(w1 * scale1), int(h1 * scale1)
#                     img1 = cv2.cvtColor(heatmap1, cv2.COLOR_BGR2RGB)
#                     img1 = Image.fromarray(img1)
#                     img1 = img1.resize((new_w1, new_h1), Image.Resampling.LANCZOS)
#                     imgtk1 = ImageTk.PhotoImage(image=img1)
#                     setattr(heatmap_label_team1, 'imgtk', imgtk1)
#                     heatmap_label_team1.config(image=imgtk1)

#                     possession_label.config(text="Team 0: 0.0%\nTeam 1: 0.0%")
#                 # --- Other Modes ---
#             else:
#                 frame = out if isinstance(out, np.ndarray) else np.zeros((480, 640, 3), dtype=np.uint8)
#                 left_frame.update_idletasks()
#                 display_w = left_frame.winfo_width()
#                 display_h = left_frame.winfo_height()
#                 h, w, _ = frame.shape
#                 scale = min(display_w / w, display_h / h)
#                 new_w, new_h = int(w * scale), int(h * scale)
#                 img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 img = Image.fromarray(img)
#                 img = img.resize((new_w, new_h), Image.Resampling.LANCZOS)
#                 imgtk = ImageTk.PhotoImage(image=img)
#                 setattr(video_label, 'imgtk', imgtk)
#                 video_label.config(image=imgtk)
#                 possession_label.config(text="Team 0: 0.0%\nTeam 1: 0.0%")

#         if thread and thread.is_alive() and not paused[0]:
#             root.after(30, update_video)


#     def add_tooltip(widget, text):
#         tooltip = tk.Toplevel(widget)
#         tooltip.withdraw()
#         tooltip.overrideredirect(True)
#         label = tk.Label(tooltip, text=text, background="#ffffe0", relief='solid', borderwidth=1)
#         label.pack()
#         def enter(event):
#             tooltip.deiconify()
#             x = event.x_root + 10
#             y = event.y_root + 10
#             tooltip.geometry(f'+{x}+{y}')
#         def leave(event):
#             tooltip.withdraw()
#         widget.bind('<Enter>', enter)
#         widget.bind('<Leave>', leave)

#     # File selection
#     tk.Button(controls, text="Select Video", command=select_file).pack(side=tk.LEFT, padx=5, pady=5)
#     file_label = tk.Label(controls, textvariable=file_name_var, bg='black', fg='white')
#     file_label.pack(side=tk.LEFT, padx=5)
#     add_tooltip(file_label, file_path_var.get())

#     # Mode selection
#     tk.Label(controls, text="Mode:", bg='black', fg='white').pack(side=tk.LEFT, padx=5)
#     mode_menu = ttk.Combobox(controls, textvariable=mode_var, values=[
#         Mode.PITCH_DETECTION, Mode.PLAYER_DETECTION, Mode.BALL_DETECTION,
#         Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION, Mode.RADAR, Mode.PASS_MAP, Mode.HEATMAP
#     ], state='readonly')
#     mode_menu.pack(side=tk.LEFT, padx=5)

#     # Device selection
#     tk.Label(controls, text="Device:", bg='black', fg='white').pack(side=tk.LEFT, padx=5)
#     device_menu = ttk.Combobox(controls, textvariable=device_var, values=['cpu', 'cuda'], state='readonly')
#     device_menu.pack(side=tk.LEFT, padx=5)

#     def remove_video():
#         stop_processing()
#         file_path_var.set("")
#         file_name_var.set("")
#         clear_video()

#     # Media controls
#     tk.Button(controls, text="Play", command=start_processing).pack(side=tk.LEFT, padx=5)
#     tk.Button(controls, text="Pause", command=pause_processing).pack(side=tk.LEFT, padx=5)
#     tk.Button(controls, text="Resume", command=resume_processing).pack(side=tk.LEFT, padx=5)
#     tk.Button(controls, text="Stop", command=stop_processing).pack(side=tk.LEFT, padx=5)
#     tk.Button(controls, text="Remove Video", command=remove_video).pack(side=tk.LEFT, padx=5)

#     root.mainloop()



import tkinter as tk
from tkinter import filedialog, messagebox
from threading import Thread, Event
from PIL import Image
import queue
import cv2
import os
import numpy as np

import torch
import ultralytics.nn.tasks

from video_modes.pitch_detection import run_pitch_detection
from video_modes.player_detection import run_player_detection
from video_modes.ball_detection import run_ball_detection
from video_modes.player_tracking import run_player_tracking
from video_modes.team_classification import run_team_classification, run_team_classification_with_possession
from video_modes.radar import run_radar
from video_modes.pass_map import run_pass_map
from video_modes.possession import PossessionTracker

# Ensure models load to the correct device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Override Ultralytics safe load to respect our device
def custom_torch_safe_load(file):
    return torch.load(file, map_location=device, weights_only=False), file
ultralytics.nn.tasks.torch_safe_load = custom_torch_safe_load

# Use CustomTkinter for modern UI
import customtkinter as ctk

class Mode:
    PITCH_DETECTION     = 'PITCH_DETECTION'
    PLAYER_DETECTION    = 'PLAYER_DETECTION'
    BALL_DETECTION      = 'BALL_DETECTION'
    PLAYER_TRACKING     = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
    RADAR               = 'RADAR'
    PASS_MAP            = 'PASS_MAP'
    HEATMAP             = 'HEATMAP'


def start_tkinter_ui():
    # Dark theme
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    root.title("Sports Video Analyzer")
    root.geometry("1200x700")

    # Left video display
    left_frame = ctk.CTkFrame(root, width=800, height=700, fg_color="black")
    left_frame.pack(side="left", fill="both", expand=True)
    left_frame.pack_propagate(False)
    video_label = ctk.CTkLabel(left_frame, text="", fg_color="black")
    video_label.pack(fill="both", expand=True)

    # Right tabs
    right_frame = ctk.CTkFrame(root, width=400, height=700)
    right_frame.pack(side="right", fill="both")
    right_frame.pack_propagate(False)
    tabview = ctk.CTkTabview(right_frame)
    tabview.pack(fill="both", expand=True, padx=5, pady=5)
    for name in ["Team Possession %","Radar","Pass Map","Heatmap"]:
        tabview.add(name)

    # Tab widgets
    possession_label = ctk.CTkLabel(tabview.tab("Team Possession %"), text="Team 0: 0.0%\nTeam 1: 0.0%", font=ctk.CTkFont(size=24, weight="bold"))
    possession_label.pack(pady=40)
    radar_label = ctk.CTkLabel(tabview.tab("Radar"), text="")
    radar_label.pack(pady=20)
    pass_map_label = ctk.CTkLabel(tabview.tab("Pass Map"), text="")
    pass_map_label.pack(pady=20)
    pass_counts_label = ctk.CTkLabel(tabview.tab("Pass Map"), text="", font=ctk.CTkFont(size=16))
    pass_counts_label.pack(pady=5)
    heat_tab = tabview.tab("Heatmap")
    ctk.CTkLabel(heat_tab, text="Team 0 Heatmap", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,0))
    heatmap_label_team0 = ctk.CTkLabel(heat_tab, text="")
    heatmap_label_team0.pack(pady=5)
    ctk.CTkLabel(heat_tab, text="Team 1 Heatmap", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,0))
    heatmap_label_team1 = ctk.CTkLabel(heat_tab, text="")
    heatmap_label_team1.pack(pady=5)

    # Controls frame
    controls = ctk.CTkFrame(left_frame, fg_color="gray20", border_width=1, border_color="gray50")
    controls.pack(fill="x", side="bottom", padx=5, pady=5)

    file_path_var = tk.StringVar()
    stop_event = None
    frame_queue = queue.Queue(maxsize=2)
    thread = None
    paused = [False]
    possession_tracker = None
    current_mode = None

    def clear_video():
        for lbl in [video_label, radar_label, pass_map_label, heatmap_label_team0, heatmap_label_team1]:
            lbl.configure(image=None)
            lbl.image = None
        possession_label.configure(text="Team 0: 0.0%\nTeam 1: 0.0%")
        pass_counts_label.configure(text="")

    def select_file():
        path = filedialog.askopenfilename(filetypes=[('Video','*.mp4 *.avi *.mov')])
        if path:
            file_path_var.set(path)
            file_label.configure(text=os.path.basename(path))
            clear_video()

    def process_video_thread(src, dev, mode, q, stop_evt, poss=None):
        try:
            if mode == Mode.PITCH_DETECTION:
                gen = run_pitch_detection(src, dev)
            elif mode == Mode.PLAYER_DETECTION:
                gen = run_player_detection(src, dev)
            elif mode == Mode.BALL_DETECTION:
                gen = run_ball_detection(src, dev)
            elif mode == Mode.PLAYER_TRACKING:
                gen = run_player_tracking(src, dev)
            elif mode == Mode.TEAM_CLASSIFICATION:
                gen = run_team_classification_with_possession(src, dev, stop_evt, poss)
            elif mode == Mode.RADAR:
                gen = run_radar(src, dev)
            elif mode == Mode.PASS_MAP:
                gen = run_pass_map(src, dev)
            elif mode == Mode.HEATMAP:
                from video_modes.heatmap import run_heatmap
                gen = run_heatmap(src, dev)
            else:
                raise NotImplementedError
            for out in gen:
                if stop_evt.is_set(): break
                frame_queue.put(out)
        except Exception as e:
            frame_queue.put((e,None))

    def start_processing():
        nonlocal thread, stop_event, possession_tracker, current_mode
        if not file_path_var.get():
            messagebox.showerror("Error","Select a video file.")
            return
        if stop_event: stop_event.set()
        stop_event = Event()
        frame_queue.queue.clear()
        paused[0] = False
        current_mode = mode_menu.get()
        possession_tracker = PossessionTracker() if current_mode==Mode.TEAM_CLASSIFICATION else None
        thread = Thread(target=process_video_thread, args=(file_path_var.get(), device_menu.get(), current_mode, frame_queue, stop_event, possession_tracker), daemon=True)
        thread.start()
        update_video()

    def pause_processing(): paused[0]=True
    def resume_processing():
        if paused[0]: paused[0]=False; update_video()
    def stop_processing():
        if stop_event: stop_event.set()
        paused[0]=False
    def remove_video():
        stop_processing(); file_path_var.set(""); file_label.configure(text=""); clear_video()

    def update_video():
        if paused[0]: return
        try:
            out = frame_queue.get_nowait()
        except queue.Empty:
            pass
        else:
            mode = current_mode
            # handle PASS_MAP
            if mode == Mode.PASS_MAP:
                if isinstance(out, tuple) and len(out)==3:
                    frame, pass_img, counts = out
                else:
                    frame = out if isinstance(out, np.ndarray) else np.zeros((480,640,3),np.uint8)
                    pass_img = np.zeros_like(frame)
                    counts = {0:0,1:0}
                # display frame
                h,w,_ = frame.shape
                dw,dh = left_frame.winfo_width(),left_frame.winfo_height()
                scale=min(dw/w,dh/h)
                img = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(img).resize((int(w*scale),int(h*scale)))
                ctk_img = ctk.CTkImage(pil_img, size=(int(w*scale),int(h*scale)))
                video_label.configure(image=ctk_img); video_label.image=ctk_img
                # pass map
                h2,w2,_ = pass_img.shape
                scale2=min(350/w2,350/h2,1.0)
                pil2 = Image.fromarray(cv2.cvtColor(pass_img,cv2.COLOR_BGR2RGB)).resize((int(w2*scale2),int(h2*scale2)))
                ctk2 = ctk.CTkImage(pil2, size=(int(w2*scale2),int(h2*scale2)))
                pass_map_label.configure(image=ctk2); pass_map_label.image=ctk2
                pass_counts_label.configure(text=f"Team 0 passes: {counts.get(0,0)}   Team 1 passes: {counts.get(1,0)}")
                possession_label.configure(text="Team 0: 0.0%\nTeam 1: 0.0%")
            # handle RADAR
            elif mode==Mode.RADAR:
                if isinstance(out,tuple) and len(out)==2:
                    frame, radar_img = out
                    h,w,_ = frame.shape
                    dw,dh=left_frame.winfo_width(),left_frame.winfo_height()
                    scale=min(dw/w,dh/h)
                    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((int(w*scale),int(h*scale)))
                    ctkf=ctk.CTkImage(pil,size=(int(w*scale),int(h*scale)))
                    video_label.configure(image=ctkf); video_label.image=ctkf
                    # radar panel
                    hr,wr,_=radar_img.shape
                    scale_r=min(350/wr,350/hr,1.0)
                    pilr=Image.fromarray(cv2.cvtColor(radar_img,cv2.COLOR_BGR2RGB)).resize((int(wr*scale_r),int(hr*scale_r)))
                    ctkr=ctk.CTkImage(pilr,size=(int(wr*scale_r),int(hr*scale_r)))
                    radar_label.configure(image=ctkr); radar_label.image=ctkr
                    possession_label.configure(text="Team 0: 0.0%\nTeam 1: 0.0%")
            # TEAM_CLASSIFICATION
            elif mode==Mode.TEAM_CLASSIFICATION:
                if isinstance(out,tuple) and len(out)==2:
                    frame,(p0,p1)=out
                    h,w,_=frame.shape
                    dw,dh=left_frame.winfo_width(),left_frame.winfo_height()
                    scale=min(dw/w,dh/h)
                    pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((int(w*scale),int(h*scale)))
                    ctkf=ctk.CTkImage(pil,size=(int(w*scale),int(h*scale)))
                    video_label.configure(image=ctkf); video_label.image=ctkf
                    possession_label.configure(text=f"Team 0: {p0:.1f}%\nTeam 1: {p1:.1f}%")
            # HEATMAP
            elif mode==Mode.HEATMAP:
                if isinstance(out,tuple) and len(out)==3:
                    frame,hm0,hm1=out
                    h,w,_=frame.shape
                    dw,dh=left_frame.winfo_width(),left_frame.winfo_height()
                    scale=min(dw/w,dh/h)
                    pil=_=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((int(w*scale),int(h*scale)))
                    ctkf=ctk.CTkImage(pil,size=(int(w*scale),int(h*scale)))
                    video_label.configure(image=ctkf); video_label.image=ctkf
                    # team0 heatmap
                    h0,w0,_=hm0.shape
                    s0=min(350/w0,350/h0,1.0)
                    pil0=Image.fromarray(cv2.cvtColor(hm0,cv2.COLOR_BGR2RGB)).resize((int(w0*s0),int(h0*s0)))
                    ctk0=ctk.CTkImage(pil0,size=(int(w0*s0),int(h0*s0)))
                    heatmap_label_team0.configure(image=ctk0); heatmap_label_team0.image=ctk0
                    # team1 heatmap
                    h1,w1,_=hm1.shape
                    s1=min(350/w1,350/h1,1.0)
                    pil1=Image.fromarray(cv2.cvtColor(hm1,cv2.COLOR_BGR2RGB)).resize((int(w1*s1),int(h1*s1)))
                    ctk1=ctk.CTkImage(pil1,size=(int(w1*s1),int(h1*s1)))
                    heatmap_label_team1.configure(image=ctk1); heatmap_label_team1.image=ctk1
                    possession_label.configure(text="Team 0: 0.0%\nTeam 1: 0.0%")
            else:
                frame=out if isinstance(out,np.ndarray) else np.zeros((480,640,3),np.uint8)
                h,w,_=frame.shape
                dw,dh=left_frame.winfo_width(),left_frame.winfo_height()
                scale=min(dw/w,dh/h)
                pil=Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((int(w*scale),int(h*scale)))
                ctkf=ctk.CTkImage(pil,size=(int(w*scale),int(h*scale)))
                video_label.configure(image=ctkf); video_label.image=ctkf
                possession_label.configure(text="Team 0: 0.0%\nTeam 1: 0.0%")
        if thread and thread.is_alive() and not paused[0]:
            root.after(30, update_video)

    # Tooltip
    def add_tooltip(w):
        tip=tk.Toplevel(w); tip.withdraw(); tip.overrideredirect(True)
        lbl=tk.Label(tip, text="", bg="#ffffe0", relief='solid', borderwidth=1); lbl.pack()
        def enter(e): lbl.config(text=file_path_var.get()); tip.deiconify(); tip.geometry(f'+{e.x_root+10}+{e.y_root+10}')
        def leave(e): tip.withdraw()
        w.bind('<Enter>',enter); w.bind('<Leave>',leave)

    # Controls grid
    ctk.CTkButton(controls,text="Select Video",command=select_file).grid(row=0,column=0,padx=5,pady=5,sticky="w")
    file_label=ctk.CTkLabel(controls,text="")
    file_label.grid(row=0,column=1,columnspan=4,padx=5,pady=5,sticky="w")
    add_tooltip(file_label)

    ctk.CTkLabel(controls,text="Mode:").grid(row=1,column=0,padx=5,pady=5,sticky="e")
    mode_menu=ctk.CTkComboBox(controls,values=[Mode.PITCH_DETECTION,Mode.PLAYER_TRACKING,Mode.TEAM_CLASSIFICATION,Mode.RADAR,Mode.PASS_MAP,Mode.HEATMAP])
    mode_menu.set(Mode.PLAYER_DETECTION)
    mode_menu.grid(row=1,column=1,padx=5,pady=5,sticky="w")

    ctk.CTkLabel(controls,text="Device:").grid(row=1,column=2,padx=5,pady=5,sticky="e")
    device_menu=ctk.CTkComboBox(controls,values=['cpu','cuda'])
    device_menu.set('cpu')
    device_menu.grid(row=1,column=3,padx=5,pady=5,sticky="w")

    # Playback buttons
    for i,(txt,cmd) in enumerate([("Play",start_processing),("Pause",pause_processing),("Resume",resume_processing),("Stop",stop_processing),("Remove Video",remove_video)]):
        ctk.CTkButton(controls,text=txt,command=cmd).grid(row=2,column=i,padx=5,pady=5)

    root.mainloop()

# import tkinter as tk
# from tkinter import filedialog, messagebox
# from threading import Thread, Event
# from PIL import Image
# import queue
# import cv2
# import os
# import numpy as np

# import torch
# import ultralytics.nn.tasks

# from video_modes.pitch_detection import run_pitch_detection
# from video_modes.player_detection import run_player_detection
# from video_modes.ball_detection import run_ball_detection
# from video_modes.player_tracking import run_player_tracking
# from video_modes.team_classification import run_team_classification, run_team_classification_with_possession
# from video_modes.radar import run_radar
# from video_modes.pass_map import run_pass_map
# from video_modes.possession import PossessionTracker

# # Ensure models load to the correct device
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Override Ultralytics safe load to respect our device
# def custom_torch_safe_load(file):
#     return torch.load(file, map_location=device, weights_only=False), file
# ultralytics.nn.tasks.torch_safe_load = custom_torch_safe_load

# # Use CustomTkinter for modern UI
# import customtkinter as ctk

# class Mode:
#     PITCH_DETECTION     = 'PITCH_DETECTION'
#     PLAYER_DETECTION    = 'PLAYER_DETECTION'
#     BALL_DETECTION      = 'BALL_DETECTION'
#     PLAYER_TRACKING     = 'PLAYER_TRACKING'
#     TEAM_CLASSIFICATION = 'TEAM_CLASSIFICATION'
#     RADAR               = 'RADAR'
#     PASS_MAP            = 'PASS_MAP'
#     HEATMAP             = 'HEATMAP'


# def start_tkinter_ui():
#     # Dark theme
#     ctk.set_appearance_mode("Dark")
#     ctk.set_default_color_theme("dark-blue")
#     root = ctk.CTk()
#     root.title("Sports Video Analyzer")
#     root.geometry("1200x700")

#     # Left video display
#     left_frame = ctk.CTkFrame(root, width=800, height=700, fg_color="black")
#     left_frame.pack(side="left", fill="both", expand=True)
#     left_frame.pack_propagate(False)
#     video_label = ctk.CTkLabel(left_frame, text="", fg_color="black")
#     video_label.pack(fill="both", expand=True)

#     # Right tabs
#     right_frame = ctk.CTkFrame(root, width=400, height=700)
#     right_frame.pack(side="right", fill="both")
#     right_frame.pack_propagate(False)
#     tabview = ctk.CTkTabview(right_frame)
#     tabview.pack(fill="both", expand=True, padx=5, pady=5)
#     for name in ["Team Possession %","Radar","Pass Map","Heatmap"]:
#         tabview.add(name)

#     # Tab widgets
#     possession_label = ctk.CTkLabel(tabview.tab("Team Possession %"), text="Team 0: 0.0%\nTeam 1: 0.0%", font=ctk.CTkFont(size=24, weight="bold"))
#     possession_label.pack(pady=40)
#     radar_label = ctk.CTkLabel(tabview.tab("Radar"), text="")
#     radar_label.pack(pady=20)
#     pass_map_label = ctk.CTkLabel(tabview.tab("Pass Map"), text="")
#     pass_map_label.pack(pady=20)
#     pass_counts_label = ctk.CTkLabel(tabview.tab("Pass Map"), text="", font=ctk.CTkFont(size=16))
#     pass_counts_label.pack(pady=5)
#     heat_tab = tabview.tab("Heatmap")
#     ctk.CTkLabel(heat_tab, text="Team 0 Heatmap", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,0))
#     heatmap_label_team0 = ctk.CTkLabel(heat_tab, text="")
#     heatmap_label_team0.pack(pady=5)
#     ctk.CTkLabel(heat_tab, text="Team 1 Heatmap", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10,0))
#     heatmap_label_team1 = ctk.CTkLabel(heat_tab, text="")
#     heatmap_label_team1.pack(pady=5)

#     # Controls frame
#     controls = ctk.CTkFrame(left_frame, fg_color="gray20", border_width=1, border_color="gray50")
#     controls.pack(fill="x", side="bottom", padx=5, pady=5)

#     file_path_var = tk.StringVar()
#     stop_event = None
#     frame_queue = queue.Queue(maxsize=2)
#     thread = None
#     paused = [False]
#     possession_tracker = None
#     current_mode = None

#     def clear_video():
#         for lbl in [video_label, radar_label, pass_map_label, heatmap_label_team0, heatmap_label_team1]:
#             lbl.configure(image=None)
#             lbl.image = None
#         possession_label.configure(text="Team 0: 0.0%\nTeam 1: 0.0%")
#         pass_counts_label.configure(text="")

#     def select_file():
#         path = filedialog.askopenfilename(filetypes=[('Video','*.mp4 *.avi *.mov')])
#         if path:
#             file_path_var.set(path)
#             file_label.configure(text=os.path.basename(path))
#             clear_video()

#     def process_video_thread(src, dev, mode, q, stop_evt, poss=None):
#         try:
#             if mode == Mode.PITCH_DETECTION:
#                 gen = run_pitch_detection(src, dev)
#             elif mode == Mode.PLAYER_TRACKING:
#                 gen = run_player_tracking(src, dev)
#             elif mode == Mode.TEAM_CLASSIFICATION:
#                 gen = run_team_classification_with_possession(src, dev, stop_evt, poss)
#             elif mode == Mode.RADAR:
#                 gen = run_radar(src, dev)
#             elif mode == Mode.PASS_MAP:
#                 gen = run_pass_map(src, dev)
#             elif mode == Mode.HEATMAP:
#                 from video_modes.heatmap import run_heatmap
#                 gen = run_heatmap(src, dev)
#             else:
#                 raise NotImplementedError
#             for out in gen:
#                 if stop_evt.is_set(): break
#                 frame_queue.put(out)
#         except Exception as e:
#             frame_queue.put((e,None))

#     def start_processing():
#         nonlocal thread, stop_event, possession_tracker, current_mode
#         if not file_path_var.get():
#             messagebox.showerror("Error","Select a video file.")
#             return
#         if stop_event: stop_event.set()
#         stop_event = Event()
#         frame_queue.queue.clear()
#         paused[0] = False
#         current_mode = mode_menu.get()
#         possession_tracker = PossessionTracker() if current_mode==Mode.TEAM_CLASSIFICATION else None
#         thread = Thread(target=process_video_thread, args=(file_path_var.get(), device_menu.get(), current_mode, frame_queue, stop_event, possession_tracker), daemon=True)
#         thread.start()
#         update_video()

#     def pause_processing(): paused[0]=True
#     def resume_processing():
#         if paused[0]: paused[0]=False; update_video()
#     def stop_processing():
#         if stop_event: stop_event.set()
#         paused[0]=False
#     def remove_video():
#         stop_processing(); file_path_var.set(""); file_label.configure(text=""); clear_video()

#     def update_video():
#         if paused[0]: return
#         try:
#             out = frame_queue.get_nowait()
#         except queue.Empty:
#             pass
#         else:
#             frame = out if not isinstance(out, tuple) else out[0]
#             h,w,_ = frame.shape
#             dw,dh = left_frame.winfo_width(), left_frame.winfo_height()
#             scale = min(dw/w, dh/h)
#             pil = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)).resize((int(w*scale),int(h*scale)))
#             img = ctk.CTkImage(pil, size=(int(w*scale),int(h*scale)))
#             video_label.configure(image=img); video_label.image=img
#         if thread and thread.is_alive() and not paused[0]:
#             root.after(30, update_video)

#     # Controls grid
#     ctk.CTkButton(controls, text="Select Video", command=select_file).grid(row=0, column=0, padx=5, pady=5, sticky="w")
#     file_label = ctk.CTkLabel(controls, text="")
#     file_label.grid(row=0, column=1, columnspan=4, padx=5, pady=5, sticky="w")

#     ctk.CTkLabel(controls, text="Mode:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
#     mode_menu = ctk.CTkComboBox(controls, values=[
#         Mode.PITCH_DETECTION, Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION,
#         Mode.RADAR, Mode.PASS_MAP, Mode.HEATMAP
#     ])
#     mode_menu.set(Mode.PITCH_DETECTION)
#     mode_menu.grid(row=1, column=1, padx=5, pady=5, sticky="w")

#     ctk.CTkLabel(controls, text="Device:").grid(row=1, column=2, padx=5, pady=5, sticky="w")
#     device_menu = ctk.CTkComboBox(controls, values=['cpu','cuda'])
#     device_menu.set('cpu')
#     device_menu.grid(row=1, column=3, padx=5, pady=5, sticky="w")

#     # Playback buttons
#     for i,(txt,cmd) in enumerate([("Play",start_processing),("Pause",pause_processing),("Resume",resume_processing),("Stop",stop_processing),("Remove",remove_video)]):
#         ctk.CTkButton(controls, text=txt, command=cmd).grid(row=2, column=i, padx=5, pady=5)

#     root.mainloop()

