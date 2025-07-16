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
from video_modes.individual_player_tracking import individual_player_tracking

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
    INDIVIDUAL_PLAYER_TRACKING = 'INDIVIDUAL_PLAYER_TRACKING'


def start_tkinter_ui():
    # Dark theme
    ctk.set_appearance_mode("Dark")
    ctk.set_default_color_theme("dark-blue")
    root = ctk.CTk()
    root.title("Sports Video Analyzer")
    root.geometry("1200x700")

    # Left video display
    left_frame = ctk.CTkFrame(root, width=1000, height=700, fg_color="black")
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
            elif mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
                # Get player ID from input field
                try:
                    player_id_text = player_id_entry.get().strip()
                    if not player_id_text:
                        messagebox.showerror("Error", "Please enter a Player ID")
                        return
                    selected_id = int(player_id_text)
                    gen = individual_player_tracking(src, dev, selected_id)
                except ValueError:
                    messagebox.showerror("Error", "Please enter a valid integer for Player ID")
                    return
                except Exception as e:
                    print(f"Error with player ID input: {e}")
                    return
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
            # INDIVIDUAL_PLAYER_TRACKING
            elif mode==Mode.INDIVIDUAL_PLAYER_TRACKING:
                # Check if this is an exception tuple
                if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], Exception):
                    print(f"Exception in individual player tracking: {out[0]}")
                    return
                elif isinstance(out, tuple) and len(out) == 2:
                    frame, annotated_frame = out
                    if (isinstance(frame, np.ndarray) and isinstance(annotated_frame, np.ndarray)):
                        # Get the actual available display area with more conservative padding
                        dw = left_frame.winfo_width() - 40  # More padding
                        dh = left_frame.winfo_height() - 40  # More padding
                        
                        # Get frame dimensions
                        h, w, _ = annotated_frame.shape
                        
                        # Calculate scale to fit the frame completely within the display area
                        scale_w = dw / w
                        scale_h = dh / h
                        scale = min(scale_w, scale_h, 0.9)  # More conservative scaling (max 90% of available space)
                        
                        # Calculate new dimensions
                        new_w = int(w * scale)
                        new_h = int(h * scale)
                        
                        # Convert and resize the annotated frame
                        img_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(img_rgb)
                        pil_img = pil_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
                        ctk_img = ctk.CTkImage(pil_img, size=(new_w, new_h))
                        
                        # Update the video label
                        video_label.configure(image=ctk_img)
                        video_label.image = ctk_img
                        possession_label.configure(text="Team 0: 0.0%\nTeam 1: 0.0%")
                    else:
                        print(f"Error: frame or annotated_frame is not a numpy array! Got: {type(frame)}, {type(annotated_frame)}")
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
    mode_menu=ctk.CTkComboBox(controls,values=[Mode.PITCH_DETECTION,Mode.PLAYER_TRACKING,Mode.TEAM_CLASSIFICATION,Mode.RADAR,Mode.PASS_MAP,Mode.HEATMAP, Mode.INDIVIDUAL_PLAYER_TRACKING])
    mode_menu.set(Mode.PLAYER_DETECTION)
    mode_menu.grid(row=1,column=1,padx=5,pady=5,sticky="w")

    ctk.CTkLabel(controls,text="Device:").grid(row=1,column=2,padx=5,pady=5,sticky="e")
    device_menu=ctk.CTkComboBox(controls,values=['cpu','cuda'])
    device_menu.set('cpu')
    device_menu.grid(row=1,column=3,padx=5,pady=5,sticky="w")

    # Player ID input for individual tracking
    ctk.CTkLabel(controls,text="Player ID:").grid(row=1,column=4,padx=5,pady=5,sticky="e")
    player_id_entry=ctk.CTkEntry(controls,placeholder_text="Enter player ID")
    player_id_entry.grid(row=1,column=5,padx=5,pady=5,sticky="w")

    # Playback buttons
    for i,(txt,cmd) in enumerate([("Play",start_processing),("Pause",pause_processing),("Resume",resume_processing),("Stop",stop_processing),("Remove Video",remove_video)]):
        ctk.CTkButton(controls,text=txt,command=cmd).grid(row=2,column=i,padx=5,pady=5)

    root.mainloop()

