# GTA V Hand Gesture Control

Control Grand Theft Auto V (GTA V) using real-time hand gestures detected via webcam and machine learning. This project uses [MediaPipe](https://mediapipe.dev/) for hand tracking, [scikit-learn](https://scikit-learn.org/) for gesture classification, and [pynput](https://pynput.readthedocs.io/) for keyboard/mouse emulation.

---

## Features

- **Real-time hand gesture recognition** via webcam
- **Custom gesture recording** and dataset creation
- **Model training** with Random Forest classifier
- **In-game control**: Map gestures to GTA V actions (move, fight, jump, enter/exit vehicle, etc.)
- **Easy extensibility** for new gestures and actions

---

## Requirements

- Python 3.8+
- OpenCV (`opencv-python`)
- MediaPipe
- NumPy
- scikit-learn
- pandas
- pynput
- joblib

Install dependencies:
```sh
pip install opencv-python mediapipe numpy scikit-learn pandas pynput joblib
```

---

## Usage

### 1. Record Gestures

Run the gesture recorder to collect training data:
```sh
python record_gesture.py
```
- Press `SPACE` to start/stop collecting samples for each gesture.
- Press `ESC` to exit.
- The dataset will be saved to `dataset.csv`.

### 2. Train the Model

Train a gesture classification model:
```sh
python trainer.py
```
- The trained model will be saved as `model.pkl`.

### 3. Control GTA V

Start the gesture control system:
```sh
python control.py
```
- Make sure GTA V is running and focused.
- Perform gestures in front of your webcam to control the game.

---

## Gesture List

Default gestures (customizable in `record_gesture.py`):

- **Left**: Move left
- **Right**: Move right
- **Fight**: Attack
- **Back**: Move backward
- **Forward**: Move forward
- **Jump**: Jump
- **Enter/Exit**: Enter or exit vehicle
- **Stop**: Stop vehicle

---

## File Overview

- `record_gesture.py` — Collects gesture data and saves to CSV.
- `trainer.py` — Trains a Random Forest model on the collected data.
- `control.py` — Runs real-time gesture recognition and controls GTA V.
- `dataset.csv` — Collected gesture data.
- `model.pkl` — Trained gesture classification model.

---

## Notes

- For best results, record gestures in consistent lighting and background.
- You can add or modify gestures by editing the `GESTURES` list in `record_gesture.py`.
- The system uses the webcam at 320x240 resolution for performance.

---

## License

This project is for educational and personal use only. Not affiliated with or endorsed by Rockstar Games.

---

## Acknowledgements

- [MediaPipe](https://mediapipe.dev/)
- [scikit-learn](https://scikit-learn.org/)
- [OpenCV](https://opencv.org/)