import cv2
import mediapipe as mp
import csv
import time
import math
import numpy as np

def flip_if_needed(landmarks):  # Check if the hand is flipped
    wrist = landmarks[0]
    middle_mcp = landmarks[5]
    if middle_mcp.y > wrist.y:
        base_y = wrist.y
        flipped = []
        for lm in landmarks:
            flipped.append(type(lm)(x=lm.x, y=base_y - (lm.y - base_y), z=lm.z))
        return flipped
    return landmarks

def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
    normalized = []
    for lm in landmarks:
        normalized.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
    return normalized

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

GESTURES = ["Left", "Right", "Fight", "Back", "Forward", "Jump", "Enter/Exit", "Stop"]
gesture_index = 0
samples_per_gesture = 400
delay_between_samples = 0.1 

csv_file = "dataset.csv"
header = [f"{coord}_{i}" for i in range(21) for coord in ("x", "y", "z")] + ["label"]
with open(csv_file, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(header)

print("[INFO] Press SPACE to start/stop collecting samples.")
print("[INFO] Press ESC to exit.")

collecting = False
collected = 0
last_sample_time = time.time()

while True:
    if gesture_index >= len(GESTURES):
        print("[INFO] All gestures collected successfully!")
        break

    gesture_name = GESTURES[gesture_index]
    ret, frame = cap.read()
    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            if collecting and time.time() - last_sample_time >= delay_between_samples:
                flipped_landmarks = flip_if_needed(hand_landmarks.landmark)
                data = normalize_landmarks(flipped_landmarks)
                data.append(gesture_name)

                with open(csv_file, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow(data)

                collected += 1
                last_sample_time = time.time()

                if collected >= samples_per_gesture:
                    print(f"[INFO] Finished collecting: {gesture_name}")
                    collected = 0
                    gesture_index += 1
                    collecting = False
                    time.sleep(1.0)

    if gesture_index < len(GESTURES):
        cv2.putText(frame, f"Gesture: {GESTURES[gesture_index]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame, f"Samples: {collected}/{samples_per_gesture}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(frame, f"Press SPACE to {'Stop' if collecting else 'Start'}", (10, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 2)

    cv2.imshow("Gesture Recorder", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC
        break
    elif key == 32: # SPACE
        collecting = not collecting

cap.release()
cv2.destroyAllWindows()
