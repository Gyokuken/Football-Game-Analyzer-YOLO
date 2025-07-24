# Football Analysis with YOLO

## Overview

This repository provides a **comprehensive, modular toolkit for football (soccer) video analysis** using state-of-the-art deep learning and computer vision.  
It enables automated extraction of tactical and performance insights from match footage, making advanced analytics accessible to coaches, analysts, researchers, and fans.

### Why is this repo important?

- **Automates tedious manual video analysis**: No more frame-by-frame annotation or manual tracking.
- **Brings advanced analytics to everyone**: Uses open-source models and tools, so anyone can run it on their own matches.
- **Modular and extensible**: Each analysis mode (detection, tracking, pass map, heatmap, etc.) is a separate module, making it easy to customize or extend.
- **Supports both Tkinter and PyQt GUIs**: Choose your preferred interface for interactive exploration.

### What does it solve?

- **Player and ball detection/tracking**: Instantly locate and track all players and the ball in every frame.
- **Team classification**: Automatically group players into teams using visual features, even without jersey color annotation.
- **Possession and pass analysis**: Quantify team possession and visualize passing networks, helping coaches understand team dynamics.
- **Heatmaps and radar views**: See where players spend most time and how they move on the pitch.
- **Individual player tracking**: Focus on a single player, measure their speed, distance covered, and movement patterns.

---

## Features

- **Player Detection & Tracking**: YOLO-based detection and ByteTrack tracking.
- **Ball Detection & Tracking**: Custom YOLO model for ball detection.
- **Team Classification**: Vision transformer (SigLIP) + UMAP clustering for robust team assignment.
- **Possession Tracking**: Real-time team possession stats.
- **Pass Map**: Detects and visualizes passes, filtering out false positives.
- **Radar View**: Mini-map showing player positions.
- **Heatmaps**: Visualizes player movement density for each team.
- **Individual Player Tracking**: Highlights a selected player, shows their speed and distance covered.
- **Multiple GUIs**: Tkinter and PyQt interfaces for interactive analysis.

---

## Installation

1. **Clone the repository:**
   ```sh
   git clone https://github.com/yourusername/Football-Analysis---YOLO.git
   cd Football-Analysis---YOLO
   ```

2. **Install dependencies:**
   ```sh
   pip install torch ultralytics transformers supervision customtkinter pyqt5 umap-learn scikit-learn joblib tqdm opencv-python
   ```

3. **Download YOLO models:**
   - Place your trained YOLO weights in the `data/` folder:
     - `football-player-detection.pt`
     - `football-ball-detection.pt`
     - `football-pitch-detection.pt`

---

## Usage

### Tkinter GUI

```sh
python main.py
```

### PyQt GUI

```sh
python main2.py
```

- Select a video file (`.mp4`, `.avi`, `.mov`).
- Choose analysis mode (Player Detection, Pass Map, Radar, etc.).
- Click **Play** to start processing.
- Switch modes at any time using the dropdown.
- For individual player tracking, enter the player ID.

---

## Project Structure

```
├── main.py                  # Tkinter UI entry point
├── main2.py                 # PyQt UI entry point
├── ui/
│   ├── tkinter_ui.py        # Tkinter GUI implementation
│   └── pyqt_ui.py           # PyQt GUI implementation
├── video_modes/
│   ├── player_detection.py
│   ├── ball_detection.py
│   ├── player_tracking.py
│   ├── team_classification.py
│   ├── pass_map.py
│   ├── radar.py
│   ├── heatmap.py
│   ├── possession.py
│   └── individual_player_tracking.py
├── sports/
│   ├── common/
│   │   ├── team.py
│   │   ├── ball.py
│   │   ├── view.py
│   ├── configs/
│   │   └── soccer.py
│   ├── annotators/
│   │   └── soccer.py
├── data/
│   ├── football-player-detection.pt
│   ├── football-ball-detection.pt
│   └── football-pitch-detection.pt
```

---

## Customization

- **Models:** Replace YOLO weights in `data/` with your own trained models for different leagues or camera setups.
- **Team Classification:** Uses SigLIP vision transformer and UMAP for robust clustering.
- **Pass Detection:** Tunable parameters in `pass_map.py` for pass distance, velocity, and filtering.

---

## Troubleshooting

- **UMAP errors:** Ensure you have `umap-learn` installed, not the legacy `umap`.
- **Model loading issues:** Check your device (CPU/GPU) and model paths.
- **GUI issues:** Make sure all dependencies are installed and use Python 3.8+.

---

## License

MIT License

---

## Credits

- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics)
- [Supervision](https://github.com/roboflow/supervision)
- [HuggingFace Transformers](https://github.com/huggingface/transformers)
- [CustomTkinter](https://github.com/TomSchimansky/CustomTkinter)
