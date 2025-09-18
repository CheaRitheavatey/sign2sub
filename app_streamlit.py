# import streamlit as st
# import cv2
# import mediapipe as mp
# import numpy as np
# import joblib
# from PIL import Image

# # ===== Load Model =====
# model = joblib.load("model/sign_rf_model.pkl")

# # ===== MediaPipe =====
# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.7,
#     min_tracking_confidence=0.7
# )

# # ===== Khmer Dictionary =====
# khmer_dict = {
#     "Again": "ម្តងទៀត",
#     "Bathroom": "បង្គន់",
#     "Eat": "បរិភោគ",
#     "Find": "ស្វែងរក",
#     "Fine": "ល្អ",
#     "Good": "ល្អ",
#     "Hello": "សួស្តី",
#     "I_Love_You": "ខ្ញុំស្រឡាញ់អ្នក",
#     "Like": "ចូលចិត្ត",
#     "Me": "ខ្ញុំ",
#     "Milk": "ទឹកដោះគោ",
#     "No": "ទេ",
#     "Please": "សូម",
#     "See_You_Later": "ជួបគ្នាវិញ",
#     "Sleep": "គេង",
#     "Talk": "និយាយ",
#     "Thank You": "អរគុណ",
#     "Understand": "យល់",
#     "Want": "ចង់",
#     "What's Up": "មានអីថ្មី",
#     "Who": "នរណា",
#     "Why": "ហេតុអ្វី",
#     "Yes": "បាទ/ចាស",
#     "You": "អ្នក"
# }

# # ===== Sidebar Controls =====
# st.sidebar.title("Settings")
# theme = st.sidebar.radio("Mode", ["Light", "Dark"])
# subtitle_size = st.sidebar.selectbox("Subtitle Size", ["Small", "Medium", "Large"])
# language = st.sidebar.radio("Language", ["English", "Khmer"])

# font_sizes = {"Small": 20, "Medium": 28, "Large": 36}
# font_size = font_sizes[subtitle_size]

# st.title("🤟 Sign Language Detection")
# frame_window = st.empty()
# prediction_placeholder = st.empty()

# # ===== Helper =====
# def extract_landmarks(hand_landmarks):
#     coords = []
#     for lm in hand_landmarks.landmark:
#         coords.extend([lm.x, lm.y, lm.z])
#     return coords

# # ===== Webcam =====
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         st.warning("Cannot access webcam.")
#         break

#     frame = cv2.flip(frame, 1)
#     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     result = hands.process(rgb_frame)

#     prediction_text = "Waiting..."

#     if result.multi_hand_landmarks:
#         for handLms in result.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
#             features = np.array(extract_landmarks(handLms)).reshape(1, -1)
#             try:
#                 pred = model.predict(features)[0]
#                 if language == "Khmer" and pred in khmer_dict:
#                     prediction_text = khmer_dict[pred]
#                 else:
#                     prediction_text = pred
#             except:
#                 prediction_text = "Error"

#     frame_window.image(frame, channels="BGR")
#     prediction_placeholder.markdown(
#         f"<h2 style='text-align:center; font-size:{font_size}px'>{prediction_text}</h2>",
#         unsafe_allow_html=True
#     )



import streamlit as st
import cv2
import numpy as np
import joblib
from cvzone.HandTrackingModule import HandDetector
from PIL import Image

# ===== Load Model =====
model = joblib.load("model/sign_rf_model.pkl")

# ===== Khmer Dictionary =====
khmer_dict = {
    "Again": "ម្តងទៀត",
    "Bathroom": "បង្គន់",
    "Eat": "បរិភោគ",
    "Find": "ស្វែងរក",
    "Fine": "ល្អ",
    "Good": "ល្អ",
    "Hello": "សួស្តី",
    "I_Love_You": "ខ្ញុំស្រឡាញ់អ្នក",
    "Like": "ចូលចិត្ត",
    "Me": "ខ្ញុំ",
    "Milk": "ទឹកដោះគោ",
    "No": "ទេ",
    "Please": "សូម",
    "See_You_Later": "ជួបគ្នាវិញ",
    "Sleep": "គេង",
    "Talk": "និយាយ",
    "Thank You": "អរគុណ",
    "Understand": "យល់",
    "Want": "ចង់",
    "What's Up": "មានអីថ្មី",
    "Who": "នរណា",
    "Why": "ហេតុអ្វី",
    "Yes": "បាទ/ចាស",
    "You": "អ្នក"
}

# ===== Sidebar Controls =====
st.sidebar.title("Settings")
theme = st.sidebar.radio("Mode", ["Light", "Dark"])
subtitle_size = st.sidebar.selectbox("Subtitle Size", ["Small", "Medium", "Large"])
language = st.sidebar.radio("Language", ["English", "Khmer"])
font_sizes = {"Small": 20, "Medium": 28, "Large": 36}
font_size = font_sizes[subtitle_size]

st.title("🤟 Sign Language Detection")
frame_window = st.empty()
prediction_placeholder = st.empty()

# ===== Hand Detector =====
detector = HandDetector(maxHands=1)

cap = cv2.VideoCapture(0)
while cap.isOpened():
    success, frame = cap.read()
    if not success:
        st.warning("Cannot access webcam.")
        break

    frame = cv2.flip(frame, 1)
    hands, img = detector.findHands(frame)  # returns list of hands and image with drawing

    prediction_text = "Waiting..."
    if hands:
        hand = hands[0]
        lmList = hand["lmList"]  # List of 21 landmark points
        features = []
        for lm in lmList:
            x, y, z = lm
            features.extend([x, y, z])
        features = np.array(features).reshape(1, -1)

        try:
            pred = model.predict(features)[0]
            if language == "Khmer" and pred in khmer_dict:
                prediction_text = khmer_dict[pred]
            else:
                prediction_text = pred
        except:
            prediction_text = "Error"

    frame_window.image(img, channels="BGR")
    prediction_placeholder.markdown(
        f"<h2 style='text-align:center; font-size:{font_size}px'>{prediction_text}</h2>",
        unsafe_allow_html=True
    )
