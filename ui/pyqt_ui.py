import sys
import os
import queue
import cv2
import numpy as np
from threading import Thread, Event
from PyQt5 import QtCore, QtGui, QtWidgets

import torch
import ultralytics.nn.tasks

from video_modes.pitch_detection import run_pitch_detection
from video_modes.player_detection import run_player_detection
from video_modes.player_tracking import run_player_tracking
from video_modes.team_classification import (
    run_team_classification,
    run_team_classification_with_possession
)
from video_modes.radar import run_radar
from video_modes.possession import PossessionTracker
from video_modes.individual_player_tracking import individual_player_tracking

# Ensure models load to the correct device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Override Ultralytics safe load to respect our device
def custom_torch_safe_load(file):
    return torch.load(file, map_location=DEVICE, weights_only=False), file
ultralytics.nn.tasks.torch_safe_load = custom_torch_safe_load

class Mode:
    PITCH_DETECTION            = 'PITCH_DETECTION'
    PLAYER_TRACKING            = 'PLAYER_TRACKING'
    TEAM_CLASSIFICATION        = 'TEAM_CLASSIFICATION'
    RADAR                      = 'RADAR'
    HEATMAP                    = 'HEATMAP'
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
            elif self.mode == Mode.PLAYER_TRACKING:
                gen = run_player_tracking(self.src, self.dev)
            elif self.mode == Mode.TEAM_CLASSIFICATION:
                gen = run_team_classification_with_possession(
                    self.src, self.dev, self.stop_event, self.extra_arg
                )
            elif self.mode == Mode.RADAR:
                gen = run_radar(self.src, self.dev)
            elif self.mode == Mode.HEATMAP:
                from video_modes.heatmap import run_heatmap
                gen = run_heatmap(self.src, self.dev)
            elif self.mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
                sid = self.extra_arg["selected_id"]
                show = self.extra_arg["show_radar"]
                gen = individual_player_tracking(self.src, self.dev, sid, show)
            else:
                raise NotImplementedError(f"Mode {self.mode} not supported")

            for out in gen:
                if self.stop_event.is_set():
                    break
                self.out_queue.put(out)
        except Exception as e:
            # On error, push a tuple so UI can handle gracefully
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

        # Side panel with tabs
        side_panel = QtWidgets.QVBoxLayout()
        main_layout.addLayout(side_panel, stretch=1)

        self.tabs = QtWidgets.QTabWidget()
        self.tabs.addTab(self._create_possession_tab(), "Team Possession %")
        self.tabs.addTab(self._create_radar_tab(), "Radar")
        self.tabs.addTab(self._create_heatmap_tab(), "Heatmap")
        side_panel.addWidget(self.tabs)

        # Controls group
        ctrl_group = QtWidgets.QGroupBox("Controls")
        ctrl_layout = QtWidgets.QGridLayout(ctrl_group)
        side_panel.addWidget(ctrl_group)

        # Row 0: file selection
        row = 0
        ctrl_layout.addWidget(QtWidgets.QLabel("Select Video:"), row, 0)
        self.file_label = QtWidgets.QLabel("No file selected")
        ctrl_layout.addWidget(self.file_label, row, 1, 1, 3)
        sel_btn = QtWidgets.QPushButton("Browse...")
        sel_btn.clicked.connect(self.select_file)
        ctrl_layout.addWidget(sel_btn, row, 4)

        # Row 1: mode, device, player ID, show radar
        row += 1
        ctrl_layout.addWidget(QtWidgets.QLabel("Mode:"), row, 0)
        self.mode_cb = QtWidgets.QComboBox()
        for m in [
            Mode.PITCH_DETECTION,
            Mode.PLAYER_TRACKING,
            Mode.TEAM_CLASSIFICATION,
            Mode.RADAR,
            Mode.HEATMAP,
            Mode.INDIVIDUAL_PLAYER_TRACKING
        ]:
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

        ctrl_layout.addWidget(QtWidgets.QLabel("Show Radar:"), row, 6)
        self.show_radar_cb = QtWidgets.QCheckBox()
        ctrl_layout.addWidget(self.show_radar_cb, row, 7)

        # Row 2: control buttons
        row += 1
        btns = [
            ("Play", self.start_processing),
            ("Pause", self.pause_processing),
            ("Resume", self.resume_processing),
            ("Stop", self.stop_processing),
            ("Clear", self.clear_video)
        ]
        for i, (txt, fn) in enumerate(btns):
            b = QtWidgets.QPushButton(txt)
            b.clicked.connect(fn)
            ctrl_layout.addWidget(b, row, i)

        # Internal state
        self.file_path = None
        self.stop_event = Event()
        self.out_queue = queue.Queue(maxsize=2)
        self.thread = None
        self.paused = False
        self.extra_arg = None

        # Frame update timer
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(30)

    def _create_possession_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        self.poss_label = QtWidgets.QLabel(
            "Team 0: 0.0% \nTeam 1: 0.0%",
            alignment=QtCore.Qt.AlignCenter
        )
        self.poss_label.setStyleSheet("font-size: 18px; font-weight: bold;")
        layout.addWidget(self.poss_label)
        return w

    def _create_radar_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        self.radar_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.radar_label)
        return w

    def _create_heatmap_tab(self):
        w = QtWidgets.QWidget()
        layout = QtWidgets.QVBoxLayout(w)
        layout.addWidget(QtWidgets.QLabel(
            "Team 0 Heatmap", alignment=QtCore.Qt.AlignCenter))
        self.heat0_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.heat0_label)
        layout.addWidget(QtWidgets.QLabel(
            "Team 1 Heatmap", alignment=QtCore.Qt.AlignCenter))
        self.heat1_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.heat1_label)
        layout.addWidget(QtWidgets.QLabel(
            "Player Heatmap", alignment=QtCore.Qt.AlignCenter))
        self.player_heat_label = QtWidgets.QLabel(alignment=QtCore.Qt.AlignCenter)
        layout.addWidget(self.player_heat_label)
        return w

    def select_file(self):
        path, _ = QtWidgets.QFileDialog.getOpenFileName(
            self, "Select Video",
            filter="Video Files (*.mp4 *.avi *.mov)"
        )
        if path:
            self.file_path = path
            self.file_label.setText(os.path.basename(path))
            self.clear_video()

    def start_processing(self):
        if not self.file_path:
            QtWidgets.QMessageBox.warning(
                self, "Error", "Please select a video file first.")
            return

        # Stop any existing thread
        self.stop_event.set()
        self.stop_event = Event()
        self.out_queue.queue.clear()
        self.paused = False

        mode = self.mode_cb.currentText()
        dev = self.dev_cb.currentText()

        if mode == Mode.TEAM_CLASSIFICATION:
            self.extra_arg = PossessionTracker()
        elif mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
            try:
                pid = int(self.player_id_le.text().strip())
            except ValueError:
                QtWidgets.QMessageBox.warning(
                    self, "Error", "Please enter a valid integer Player ID.")
                return
            show = self.show_radar_cb.isChecked()
            self.extra_arg = {"selected_id": pid, "show_radar": show}
        else:
            self.extra_arg = None

        self.thread = VideoProcessorThread(
            self.file_path, dev, mode,
            self.out_queue, self.stop_event,
            self.extra_arg
        )
        self.thread.start()

    def pause_processing(self):
        self.paused = True

    def resume_processing(self):
        self.paused = False

    def stop_processing(self):
        self.stop_event.set()
        self.paused = False

    def clear_video(self):
        for lbl in [
            self.video_label,
            self.radar_label,
            self.heat0_label,
            self.heat1_label
        ]:
            lbl.clear()
        self.poss_label.setText("Team 0: 0.0% \nTeam 1: 0.0%")

    def update_frame(self):
        if self.paused:
            return

        try:
            out = self.out_queue.get_nowait()
        except queue.Empty:
            return

        mode = self.mode_cb.currentText()

        frame = None
        radar_img = None
        pass_img = None
        hm0 = hm1 = None
        counts = {}
        p0 = p1 = 0

        if mode == Mode.RADAR:
            frame, radar_img = out if isinstance(out, tuple) and len(out) == 2 else (None, None)
        elif mode == Mode.TEAM_CLASSIFICATION:
            frame, (p0, p1) = out if isinstance(out, tuple) and len(out) == 2 else (None, (0, 0))
        elif mode == Mode.HEATMAP:
            frame, hm0, hm1 = out if isinstance(out, tuple) and len(out) == 3 else (None, None, None)
        elif mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
            if isinstance(out, tuple):
                if len(out) == 2:
                    frame, ann = out
                    player_heatmap = None
                elif len(out) == 3:
                    frame, ann, player_heatmap = out
                if ann is not None:
                    frame = ann
        else:
            frame = out if isinstance(out, np.ndarray) else None

        # display main video frame
        if frame is not None:
            self._display_image(frame, self.video_label)
        # display radar if present
        if radar_img is not None:
            self._display_image(radar_img, self.radar_label)
        # display pass map
        if pass_img is not None:
            self._display_image(pass_img, self.pass_map_label)
            self.pass_counts_label.setText(f"Team 0: {counts.get(0,0)}   Team 1: {counts.get(1,0)}")
        # update team possession
        if mode == Mode.TEAM_CLASSIFICATION:
            self.poss_label.setText(f"Team 0: {p0:.1f}%\nTeam 1: {p1:.1f}%")
        # display heatmaps
        if mode == Mode.INDIVIDUAL_PLAYER_TRACKING:
            # Hide team heatmaps, show only player heatmap
            self.heat0_label.clear()
            self.heat1_label.clear()
            if 'player_heatmap' in locals() and player_heatmap is not None:
                print("UI: Displaying player heatmap", type(player_heatmap), getattr(player_heatmap, 'shape', None), getattr(player_heatmap, 'dtype', None))
                self._display_image(player_heatmap, self.player_heat_label)
            else:
                print("UI: No player heatmap to display")
                self.player_heat_label.clear()
        else:
            if hm0 is not None:
                self._display_image(hm0, self.heat0_label)
            if hm1 is not None:
                self._display_image(hm1, self.heat1_label)
            self.player_heat_label.clear()

    def _display_image(self, img: np.ndarray, widget: QtWidgets.QLabel):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(
            rgb.data, w, h, bytes_per_line,
            QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qt_img)
        widget.setPixmap(
            pix.scaled(
                widget.size(),
                QtCore.Qt.KeepAspectRatio,
                QtCore.Qt.SmoothTransformation
            )
        )