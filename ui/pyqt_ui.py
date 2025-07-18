# import sys
# import os
# import queue
# import cv2
# import numpy as np
# from threading import Thread, Event
# from PIL import Image
# from PyQt5 import QtCore, QtGui, QtWidgets

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
# from video_modes.individual_player_tracking import individual_player_tracking

# # Ensure models load to the correct device
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Override Ultralytics safe load to respect our device
# def custom_torch_safe_load(file):
#     return torch.load(file, map_location=DEVICE, weights_only=False), file
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
#     INDIVIDUAL_PLAYER_TRACKING = 'INDIVIDUAL_PLAYER_TRACKING'

# class VideoProcessorThread(Thread):
#     def __init__(self, src, dev, mode, out_queue, stop_event, poss=None):
#         super().__init__(daemon=True)
#         self.src = src
#         self.dev = dev
#         self.mode = mode
#         self.out_queue = out_queue
#         self.stop_event = stop_event
#         self.poss = poss

#     def run(self):
#         try:
#             # select generator based on mode
#             if self.mode == Mode.PITCH_DETECTION:
#                 gen = run_pitch_detection(self.src, self.dev)
#             elif self.mode == Mode.PLAYER_DETECTION:
#                 gen = run_player_detection(self.src, self.dev)
#             elif self.mode == Mode.BALL_DETECTION:
#                 gen = run_ball_detection(self.src, self.dev)
#             elif self.mode == Mode.PLAYER_TRACKING:
#                 gen = run_player_tracking(self.src, self.dev)
#             elif self.mode == Mode.TEAM_CLASSIFICATION:
#                 gen = run_team_classification_with_possession(self.src, self.dev, self.stop_event, self.poss)
#             elif self.mode == Mode.RADAR:
#                 gen = run_radar(self.src, self.dev)
#             elif self.mode == Mode.PASS_MAP:
#                 gen = run_pass_map(self.src, self.dev)
#             elif self.mode == Mode.HEATMAP:
#                 from video_modes.heatmap import run_heatmap
#                 gen = run_heatmap(self.src, self.dev)
#             elif self.mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
#                 gen = individual_player_tracking(self.src, self.dev, self.poss)
#             else:
#                 raise NotImplementedError

#             for out in gen:
#                 if self.stop_event.is_set():
#                     break
#                 self.out_queue.put(out)
#         except Exception as e:
#             self.out_queue.put((e, None))

# class MainWindow(QtWidgets.QMainWindow):
#     def __init__(self):
#         super().__init__()
#         self.setWindowTitle("Sports Video Analyzer")
#         self.resize(1200, 700)

#         # Widgets
#         central = QtWidgets.QWidget()
#         self.setCentralWidget(central)
#         main_layout = QtWidgets.QHBoxLayout(central)

#         # Video display area
#         self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
#         self.video_label.setStyleSheet("background-color: black;")
#         main_layout.addWidget(self.video_label, stretch=3)

#         # Right panel for metrics/tabs
#         side_panel = QtWidgets.QVBoxLayout()
#         main_layout.addLayout(side_panel, stretch=1)

#         self.tabs = QtWidgets.QTabWidget()
#         self.tabs.addTab(self._create_possession_tab(), "Team Possession %")
#         self.tabs.addTab(self._create_radar_tab(), "Radar")
#         self.tabs.addTab(self._create_pass_map_tab(), "Pass Map")
#         self.tabs.addTab(self._create_heatmap_tab(), "Heatmap")
#         side_panel.addWidget(self.tabs)

#         # Controls
#         ctrl_group = QtWidgets.QGroupBox("Controls")
#         ctrl_layout = QtWidgets.QGridLayout(ctrl_group)
#         side_panel.addWidget(ctrl_group)

#         row = 0
#         ctrl_layout.addWidget(QtWidgets.QLabel("Select Video:"), row, 0)
#         self.file_label = QtWidgets.QLabel("No file selected")
#         ctrl_layout.addWidget(self.file_label, row, 1, 1, 3)
#         sel_btn = QtWidgets.QPushButton("Browse...")
#         sel_btn.clicked.connect(self.select_file)
#         ctrl_layout.addWidget(sel_btn, row, 4)

#         row += 1
#         ctrl_layout.addWidget(QtWidgets.QLabel("Mode:"), row, 0)
#         self.mode_cb = QtWidgets.QComboBox()
#         for m in [Mode.PITCH_DETECTION, Mode.PLAYER_DETECTION, Mode.BALL_DETECTION,
#                   Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION, Mode.RADAR,
#                   Mode.PASS_MAP, Mode.HEATMAP, Mode.INDIVIDUAL_PLAYER_TRACKING]:
#             self.mode_cb.addItem(m)
#         ctrl_layout.addWidget(self.mode_cb, row, 1)

#         ctrl_layout.addWidget(QtWidgets.QLabel("Device:"), row, 2)
#         self.dev_cb = QtWidgets.QComboBox()
#         self.dev_cb.addItems(['cpu', 'cuda'])
#         ctrl_layout.addWidget(self.dev_cb, row, 3)

#         ctrl_layout.addWidget(QtWidgets.QLabel("Player ID:"), row, 4)
#         self.player_id_le = QtWidgets.QLineEdit()
#         self.player_id_le.setPlaceholderText("Enter integer")
#         ctrl_layout.addWidget(self.player_id_le, row, 5)

#         row += 1
#         btns = [("Play", self.start_processing), ("Pause", self.pause_processing),
#                 ("Resume", self.resume_processing), ("Stop", self.stop_processing),
#                 ("Clear", self.clear_video)]
#         for i, (txt, fn) in enumerate(btns):
#             b = QtWidgets.QPushButton(txt)
#             b.clicked.connect(fn)
#             ctrl_layout.addWidget(b, row, i)

#         # Internal state
#         self.file_path = None
#         self.stop_event = Event()
#         self.out_queue = queue.Queue(maxsize=2)
#         self.thread = None
#         self.paused = False
#         self.poss_tracker = None

#         # Timer for updating GUI
#         self.timer = QtCore.QTimer()
#         self.timer.timeout.connect(self.update_frame)
#         self.timer.start(30)

#     def _create_possession_tab(self):
#         w = QtWidgets.QWidget()
#         layout = QtWidgets.QVBoxLayout(w)
#         self.poss_label = QtWidgets.QLabel("Team 0: 0.0% \nTeam 1: 0.0%", alignment=QtCore.Qt.AlignCenter)
#         self.poss_label.setStyleSheet("font-size: 18px; font-weight: bold;")
#         layout.addWidget(self.poss_label)
#         return w

#     def _create_radar_tab(self):
#         w = QtWidgets.QWidget()
#         layout = QtWidgets.QVBoxLayout(w)
#         self.radar_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
#         layout.addWidget(self.radar_label)
#         return w

#     def _create_pass_map_tab(self):
#         w = QtWidgets.QWidget()
#         layout = QtWidgets.QVBoxLayout(w)
#         self.pass_map_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
#         layout.addWidget(self.pass_map_label)
#         self.pass_counts_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
#         layout.addWidget(self.pass_counts_label)
#         return w

#     def _create_heatmap_tab(self):
#         w = QtWidgets.QWidget()
#         layout = QtWidgets.QVBoxLayout(w)
#         layout.addWidget(QtWidgets.QLabel("Team 0 Heatmap", alignment=QtCore.Qt.AlignCenter))
#         self.heat0_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
#         layout.addWidget(self.heat0_label)
#         layout.addWidget(QtWidgets.QLabel("Team 1 Heatmap", alignment=QtCore.Qt.AlignCenter))
#         self.heat1_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
#         layout.addWidget(self.heat1_label)
#         return w

#     def select_file(self):
#         path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video", filter="Video Files (*.mp4 *.avi *.mov)")
#         if path:
#             self.file_path = path
#             self.file_label.setText(os.path.basename(path))
#             self.clear_video()

#     def start_processing(self):
#         if not self.file_path:
#             QtWidgets.QMessageBox.warning(self, "Error", "Please select a video file first.")
#             return
#         # stop previous
#         self.stop_event.set()
#         self.stop_event = Event()
#         self.out_queue.queue.clear()
#         self.paused = False
#         mode = self.mode_cb.currentText()
#         dev = self.dev_cb.currentText()
#         self.poss_tracker = PossessionTracker() if mode == Mode.TEAM_CLASSIFICATION else None
#         self.thread = VideoProcessorThread(self.file_path, dev, mode, self.out_queue, self.stop_event, self.poss_tracker)
#         self.thread.start()

#     def pause_processing(self):
#         self.paused = True

#     def resume_processing(self):
#         self.paused = False

#     def stop_processing(self):
#         self.stop_event.set()
#         self.paused = False

#     def clear_video(self):
#         for lbl in [self.video_label, self.radar_label, self.pass_map_label, self.heat0_label, self.heat1_label]:
#             lbl.clear()
#         self.poss_label.setText("Team 0: 0.0% \nTeam 1: 0.0%")
#         self.pass_counts_label.clear()

#     def update_frame(self):
#         if self.paused:
#             return
#         try:
#             out = self.out_queue.get_nowait()
#         except queue.Empty:
#             return

#         mode = self.mode_cb.currentText()
#         # Process output based on mode
#         # Similar to tkinter implementation, handle each tuple format
#         frame = None
#         if mode == Mode.PASS_MAP:
#             # out: (frame, pass_img, counts)
#             frame, pass_img, counts = out if isinstance(out, tuple) and len(out) == 3 else (None, None, {})
#         elif mode == Mode.RADAR:
#             frame, radar_img = out if isinstance(out, tuple) and len(out) == 2 else (None, None)
#         elif mode == Mode.TEAM_CLASSIFICATION:
#             frame, (p0, p1) = out if isinstance(out, tuple) and len(out) == 2 else (None, (0,0))
#         elif mode == Mode.HEATMAP:
#             frame, hm0, hm1 = out if isinstance(out, tuple) and len(out) == 3 else (None, None, None)
#         elif mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
#             frame, ann = out if isinstance(out, tuple) and len(out) == 2 else (None, None)
#         else:
#             frame = out if isinstance(out, np.ndarray) else None

#         # Display frame
#         if frame is not None:
#             self._display_image(frame, self.video_label)

#         # Side panel updates
#         if mode == Mode.PASS_MAP and pass_img is not None:
#             self._display_image(pass_img, self.pass_map_label)
#             self.pass_counts_label.setText(f"Team 0: {counts.get(0,0)}   Team 1: {counts.get(1,0)}")
#         if mode == Mode.RADAR and radar_img is not None:
#             self._display_image(radar_img, self.radar_label)
#         if mode == Mode.TEAM_CLASSIFICATION:
#             self.poss_label.setText(f"Team 0: {p0:.1f}%\nTeam 1: {p1:.1f}%")
#         if mode == Mode.HEATMAP:
#             if hm0 is not None: self._display_image(hm0, self.heat0_label)
#             if hm1 is not None: self._display_image(hm1, self.heat1_label)

#     def _display_image(self, img: np.ndarray, widget: QtWidgets.QLabel):
#         # Convert BGR to RGB
#         rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#         h, w, ch = rgb.shape
#         bytes_per_line = ch * w
#         qt_img = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
#         pix = QtGui.QPixmap.fromImage(qt_img)
#         widget.setPixmap(pix.scaled(widget.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))

import sys
import os
import queue
import cv2
import numpy as np
from threading import Thread, Event
from PIL import Image
from PyQt5 import QtCore, QtGui, QtWidgets

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
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Override Ultralytics safe load to respect our device
def custom_torch_safe_load(file):
    return torch.load(file, map_location=DEVICE, weights_only=False), file
ultralytics.nn.tasks.torch_safe_load = custom_torch_safe_load

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

class VideoProcessorThread(Thread):
    def __init__(self, src, dev, mode, out_queue, stop_event, extra_arg=None):
        super().__init__(daemon=True)
        self.src = src
        self.dev = dev
        self.mode = mode
        self.out_queue = out_queue
        self.stop_event = stop_event
        self.extra_arg = extra_arg

    def run(self):
        try:
            if self.mode == Mode.PITCH_DETECTION:
                gen = run_pitch_detection(self.src, self.dev)
            elif self.mode == Mode.PLAYER_DETECTION:
                gen = run_player_detection(self.src, self.dev)
            elif self.mode == Mode.BALL_DETECTION:
                gen = run_ball_detection(self.src, self.dev)
            elif self.mode == Mode.PLAYER_TRACKING:
                gen = run_player_tracking(self.src, self.dev)
            elif self.mode == Mode.TEAM_CLASSIFICATION:
                gen = run_team_classification_with_possession(self.src, self.dev, self.stop_event, self.extra_arg)
            elif self.mode == Mode.RADAR:
                gen = run_radar(self.src, self.dev)
            elif self.mode == Mode.PASS_MAP:
                gen = run_pass_map(self.src, self.dev)
            elif self.mode == Mode.HEATMAP:
                from video_modes.heatmap import run_heatmap
                gen = run_heatmap(self.src, self.dev)
            elif self.mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
                # extra_arg here is selected_id
                gen = individual_player_tracking(self.src, self.dev, self.extra_arg)
            else:
                raise NotImplementedError(f"Mode {self.mode} not supported")

            for out in gen:
                if self.stop_event.is_set():
                    break
                self.out_queue.put(out)
        except Exception as e:
            self.out_queue.put((e, None))

class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sports Video Analyzer")
        self.resize(1200, 700)

        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Video display
        self.video_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        main_layout.addWidget(self.video_label, stretch=3)

        # Side panel
        side_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(side_panel, stretch=1)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._create_possession_tab(), "Team Possession %")
        self.tabs.addTab(self._create_radar_tab(), "Radar")
        self.tabs.addTab(self._create_pass_map_tab(), "Pass Map")
        self.tabs.addTab(self._create_heatmap_tab(), "Heatmap")
        side_panel.addWidget(self.tabs)

        # Controls
        ctrl_group = QtWidgets.QGroupBox("Controls")
        ctrl_layout = QtWidgets.QGridLayout(ctrl_group)
        side_panel.addWidget(ctrl_group)

        row = 0
        ctrl_layout.addWidget(QtWidgets.QLabel("Select Video:"), row, 0)
        self.file_label = QtWidgets.QLabel("No file selected")
        ctrl_layout.addWidget(self.file_label, row, 1, 1, 3)
        sel_btn = QtWidgets.QPushButton("Browse...")
        sel_btn.clicked.connect(self.select_file)
        ctrl_layout.addWidget(sel_btn, row, 4)

        row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("Mode:"), row, 0)
        self.mode_cb = QtWidgets.QComboBox()
        for m in [Mode.PITCH_DETECTION, Mode.PLAYER_DETECTION, Mode.BALL_DETECTION,
                  Mode.PLAYER_TRACKING, Mode.TEAM_CLASSIFICATION, Mode.RADAR,
                  Mode.PASS_MAP, Mode.HEATMAP, Mode.INDIVIDUAL_PLAYER_TRACKING]:
            self.mode_cb.addItem(m)
        ctrl_layout.addWidget(self.mode_cb, row, 1)

        ctrl_layout.addWidget(QtWidgets.QLabel("Device:"), row, 2)
        self.dev_cb = QtWidgets.QComboBox()
        self.dev_cb.addItems(['cpu', 'cuda'])
        ctrl_layout.addWidget(self.dev_cb, row, 3)

        ctrl_layout.addWidget(QtWidgets.QLabel("Player ID:"), row, 4)
        self.player_id_le = QtWidgets.QLineEdit()
        self.player_id_le.setPlaceholderText("Enter integer")
        ctrl_layout.addWidget(self.player_id_le, row, 5)

        row += 1
        btns = [("Play", self.start_processing), ("Pause", self.pause_processing),
                ("Resume", self.resume_processing), ("Stop", self.stop_processing),
                ("Clear", self.clear_video)]
        for i, (txt, fn) in enumerate(btns):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            ctrl_layout.addWidget(b, row, i)

        # State
        self.file_path = None
        self.stop_event = Event()
        self.out_queue = queue.Queue(maxsize=2)
        self.thread = None
        self.paused = False
        self.extra_arg = None

        # Timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def _create_possession_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        self.poss_label = QtWidgets.QLabel("Team 0: 0.0% \nTeam 1: 0.0%", alignment=QtCore.Qt.AlignCenter)
        self.poss_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.poss_label)
        return w

    def _create_radar_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        self.radar_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.radar_label)
        return w

    def _create_pass_map_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        self.pass_map_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.pass_map_label)
        self.pass_counts_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.pass_counts_label)
        return w

    def _create_heatmap_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(QtWidgets.QLabel("Team 0 Heatmap", alignment=QtCore.Qt.AlignCenter))
        self.heat0_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.heat0_label)
        layout.addWidget(QtWidgets.QLabel("Team 1 Heatmap", alignment=QtCore.Qt.AlignCenter))
        self.heat1_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.heat1_label)
        return w

    def select_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Select Video", filter="Video Files (*.mp4 *.avi *.mov)")
        if path:
            self.file_path = path
            self.file_label.setText(os.path.basename(path))
            self.clear_video()

    def start_processing(self):
        if not self.file_path:
            QtWidgets.QMessageBox.warning(self, "Error", "Please select a video file first.")
            return
        # Stop existing
        self.stop_event.set()
        self.stop_event = Event()
        self.out_queue.queue.clear()
        self.paused = False

        mode = self.mode_cb.currentText()
        dev = self.dev_cb.currentText()

        # Determine extra_arg: possession tracker or selected_id
        if mode == Mode.TEAM_CLASSIFICATION:
            self.extra_arg = PossessionTracker()
        elif mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
            try:
                self.extra_arg = int(self.player_id_le.text().strip())
            except ValueError:
                QtWidgets.QMessageBox.warning(self, "Error", "Please enter a valid integer Player ID.")
                return
        else:
            self.extra_arg = None

        self.thread = VideoProcessorThread(self.file_path, dev, mode, self.out_queue, self.stop_event, self.extra_arg)
        self.thread.start()

    def pause_processing(self):
        self.paused = True

    def resume_processing(self):
        self.paused = False

    def stop_processing(self):
        self.stop_event.set()
        self.paused = False

    def clear_video(self):
        for lbl in [self.video_label, self.radar_label, self.pass_map_label, self.heat0_label, self.heat1_label]:
            lbl.clear()
        self.poss_label.setText("Team 0: 0.0% \nTeam 1: 0.0%")
        self.pass_counts_label.clear()

    def update_frame(self):
        if self.paused:
            return
        try:
            out = self.out_queue.get_nowait()
        except queue.Empty:
            return

        mode = self.mode_cb.currentText()
        frame = None
        pass_img = radar_img = hm0 = hm1 = None
        counts = {}
        p0 = p1 = 0

        if mode == Mode.PASS_MAP:
            frame, pass_img, counts = out if isinstance(out, tuple) and len(out)==3 else (None, None, {})
        elif mode == Mode.RADAR:
            frame, radar_img = out if isinstance(out, tuple) and len(out)==2 else (None, None)
        elif mode == Mode.TEAM_CLASSIFICATION:
            frame, (p0, p1) = out if isinstance(out, tuple) and len(out)==2 else (None, (0,0))
        elif mode == Mode.HEATMAP:
            frame, hm0, hm1 = out if isinstance(out, tuple) and len(out)==3 else (None, None, None)
        elif mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
            frame, ann = out if isinstance(out, tuple) and len(out)==2 else (None, None)
            if ann is not None:
                frame = ann
        else:
            frame = out if isinstance(out, np.ndarray) else None

        if frame is not None:
            self._display_image(frame, self.video_label)
        if pass_img is not None:
            self._display_image(pass_img, self.pass_map_label)
            self.pass_counts_label.setText(f"Team 0: {counts.get(0,0)}   Team 1: {counts.get(1,0)}")
        if radar_img is not None:
            self._display_image(radar_img, self.radar_label)
        if mode == Mode.TEAM_CLASSIFICATION:
            self.poss_label.setText(f"Team 0: {p0:.1f}%\nTeam 1: {p1:.1f}%")
        if hm0 is not None:
            self._display_image(hm0, self.heat0_label)
        if hm1 is not None:
            self._display_image(hm1, self.heat1_label)

    def _display_image(self, img: np.ndarray, widget: QtWidgets.QLabel):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        pix = QtGui.QPixmap.fromImage(qt_img)
        widget.setPixmap(pix.scaled(widget.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation))
