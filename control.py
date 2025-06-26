import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
import time
import cv2
import mediapipe as mp
import math
import numpy as np 
import joblib
from collections import deque, Counter
from pynput.keyboard import Key, Controller as KeyboardController
from pynput.mouse import Button, Controller as MouseController

keyboard = KeyboardController()
mouse = MouseController()

gesture_buffer = deque(maxlen=5) 
current_action = "Idle"
last_update_time = 0
display_duration = 1.0 

def flip_if_needed(landmarks):
    wrist = landmarks[0]
    middle_mcp = landmarks[9]
    if middle_mcp.y > wrist.y:
        base_y = wrist.y
        flipped = []
        for lm in landmarks:
            flipped.append(type(lm)(x=lm.x, y=base_y - (lm.y - base_y), z=lm.z))
        return flipped
    return landmarks

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1, model_complexity=0)
mp_draw = mp.solutions.drawing_utils

model = joblib.load("model.pkl")

in_vehicle = False  

def perform_action(gesture):
    global in_vehicle
    keyboard.release('w')
    keyboard.release('a')
    keyboard.release('s')
    keyboard.release('d')
    keyboard.release('f')
    keyboard.release(Key.space)
    mouse.release(Button.left)

    if gesture == "Enter/Exit":
        in_vehicle = not in_vehicle
        keyboard.press('f')

    elif gesture == "Left":
        if in_vehicle:
            keyboard.press('a')
            keyboard.press('w')
        else:
            keyboard.press('a')

    elif gesture == "Right":
        if in_vehicle:
            keyboard.press('d')
            keyboard.press('w')
        else:
            keyboard.press('d')

    elif gesture == "Forward":
        keyboard.press('w')

    elif gesture == "Back":
        keyboard.press('s')

    elif gesture == "Jump":
        keyboard.press(Key.space)

    elif gesture == "Fight":
        mouse.press(Button.left)

    elif gesture == "Stop":
        if in_vehicle:
            keyboard.press(Key.space)

def get_hand_size(landmarks, width, height): # distance between wrist and index finger
    x1, y1 = int(landmarks[0].x * width), int(landmarks[0].y * height)
    x2, y2 = int(landmarks[8].x * width), int(landmarks[8].y * height)
    return math.hypot(x2 - x1, y2 - y1)

def get_depth_size(landmarks, width, height): 
    x1, y1 = int(landmarks[0].x * width), int(landmarks[0].y * height) # wrist
    x2, y2 = int(landmarks[8].x * width), int(landmarks[8].y * height) # index finger
    return math.hypot(x2 - x1, y2 - y1)

cap = cv2.VideoCapture(0) 
cap.set(3, 320)
cap.set(4, 240)

while True:
    success, frame = cap.read()
    if not success:
        continue

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
        
            def normalize_landmarks(landmarks):
                base_x, base_y, base_z = landmarks[0].x, landmarks[0].y, landmarks[0].z
                normalized = []
                for lm in landmarks:
                    normalized.extend([lm.x - base_x, lm.y - base_y, lm.z - base_z])
                return normalized

            features = normalize_landmarks(hand_landmarks.landmark)
 
            if len(features) == 63:

                gesture = model.predict([features])[0]
                gesture_buffer.append(gesture)

                smoothed_gesture = Counter(gesture_buffer).most_common(1)[0][0]
                if smoothed_gesture != current_action:
                    perform_action(smoothed_gesture)
                    current_action = smoothed_gesture
                    last_update_time = time.time()

                if get_depth_size(hand_landmarks.landmark, w, h) > 170: # Hand distance sensitivity for shift key
                    keyboard.press(Key.shift)
                else:
                    keyboard.release(Key.shift)

                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                text_to_show = f"Action: {current_action}"

                cv2.putText(frame, text_to_show, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Hand Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
