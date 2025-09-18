# # import tkinter as tk
# # from tkinter import ttk
# # import cv2
# # from PIL import Image, ImageTk
# # import threading
# # import queue
# # from gui.main_window import MainWindow

# # def main():
# #     root = tk.Tk()
# #     app = MainWindow(root)
# #     root.mainloop()

# # if __name__ == "__main__":
# #     main()
    
# import numpy as np
# import mediapipe as mp
# import joblib

# class GestureDetector:
#     def __init__(self):
#         self.confidence_threshold = 0.7
#         self.space = 20
#         self.imgSize = 300

#         # Load RandomForest model (trained from CSV)
#         try:
#             self.model = joblib.load("model/sign_rf_model.pkl")  # 👈 trained model path
#             self.model_loaded = True
#             print("✅ RandomForest model loaded")
#         except Exception as e:
#             print(f"Error loading RandomForest model: {e}")
#             self.model_loaded = False

#         # Mediapipe Hands
#         self.mp_hands = mp.solutions.hands
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=1,
#             min_detection_confidence=0.5,
#             min_tracking_confidence=0.5
#         )
#         self.mp_draw = mp.solutions.drawing_utils

#     def set_confidence_threshold(self, threshold):
#         self.confidence_threshold = threshold

#     def get_gesture_text(self, label, language="english"):
#         """Return gesture text in specified language"""
#         # For now, labels are already strings (e.g., "Hello", "ThankYou")
#         return label

#     def process_frame(self, frame):
#         if not self.model_loaded:
#             return frame, None, 0.0

#         rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = self.hands.process(rgb)

#         if results.multi_hand_landmarks:
#             for handLms in results.multi_hand_landmarks:
#                 row = []
#                 for lm in handLms.landmark:
#                     row += [lm.x, lm.y, lm.z]
#                 row = np.array(row).reshape(1, -1)

#                 # Predict gesture
#                 pred = self.model.predict(row)[0]
#                 # NOTE: scikit-learn RandomForest doesn't give probabilities by default
#                 # If you need confidence, use predict_proba
#                 proba = self.model.predict_proba(row).max()

#                 # Draw landmarks
#                 self.mp_draw.draw_landmarks(frame, handLms, self.mp_hands.HAND_CONNECTIONS)

#                 return frame, pred, proba

#         return frame, None, 0.0

# GestureDetector()



import cv2
import mediapipe as mp
import numpy as np
import joblib
import tkinter as tk
from PIL import Image, ImageTk

# ===== Load Trained Model =====
model = joblib.load("model/sign_rf_model.pkl")  # RandomForest model trained on sign_data.csv

# ===== MediaPipe Setup =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)

# ===== GUI Setup =====
root = tk.Tk()
root.title("Sign Language Detection")
root.geometry("900x700")

label_result = tk.Label(root, text="Prediction: None", font=("Arial", 24))
label_result.pack(pady=10)

video_label = tk.Label(root)
video_label.pack()

cap = cv2.VideoCapture(0)  # Webcam

def extract_landmarks(hand_landmarks):
    """
    Extract x, y, z coordinates for each of the 21 hand landmarks
    Return as a flattened list [x1,y1,z1, x2,y2,z2, ...]
    """
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

def update_frame():
    ret, frame = cap.read()
    if not ret:
        root.after(10, update_frame)
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

            # Extract features & Predict
            features = np.array(extract_landmarks(handLms)).reshape(1, -1)
            try:
                prediction = model.predict(features)
                label_result.config(text=f"Prediction: {prediction[0]}")
            except Exception as e:
                label_result.config(text="Prediction: Error")

    # Convert frame for Tkinter
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    root.after(10, update_frame)

def on_close():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
update_frame()
root.mainloop()
