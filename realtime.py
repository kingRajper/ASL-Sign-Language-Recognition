import os
import logging
import tensorflow as tf
import absl.logging

# Disable most TensorFlow logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'            # 3 = ERROR only
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'           # Turn off oneDNN logs

# Set Python logging levels
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('absl').setLevel(logging.ERROR)
absl.logging.set_verbosity(absl.logging.ERROR)


import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import load_model



# --- CONFIG ---
model_path = 'asl_mobilenet_model.h5'
img_size = 224
labels = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M',
          'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z',
          'DELETE', 'NOTHING', 'SPACE']

# --- LOAD MODEL ---
model = load_model(model_path)

# --- INIT MEDIAPIPE HAND DETECTOR ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# --- START CAMERA ---
cap = cv2.VideoCapture(0)

print("[INFO] Press 'q' to quit.")
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural view
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    # Detect hands
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Get bounding box from landmarks
            x_coords = [lm.x for lm in hand_landmarks.landmark]
            y_coords = [lm.y for lm in hand_landmarks.landmark]
            x_min = int(min(x_coords) * w) - 20
            y_min = int(min(y_coords) * h) - 20
            x_max = int(max(x_coords) * w) + 20
            y_max = int(max(y_coords) * h) + 20

            # Clamp values
            x_min = max(x_min, 0)
            y_min = max(y_min, 0)
            x_max = min(x_max, w)
            y_max = min(y_max, h)

            # Crop and preprocess
            hand_img = frame[y_min:y_max, x_min:x_max]
            if hand_img.size == 0:
                continue  # Skip empty frame

            hand_img = cv2.resize(hand_img, (img_size, img_size))
            hand_img = hand_img.astype('float32') / 255.0
            hand_img = np.expand_dims(hand_img, axis=0)

            # Predict
            preds = model.predict(hand_img)
            pred_label = labels[np.argmax(preds)]

            # Draw bounding box and prediction
            cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
            cv2.putText(frame, f'{pred_label}', (x_min, y_min - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("ASL Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
