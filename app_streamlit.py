# # import cv2
# # import mediapipe as mp
# # import numpy as np
# # import joblib
# # from PIL import Image, ImageTk
# # import customtkinter as ctk

# # # ===== Load Model =====
# # model = joblib.load("model/sign_rf_model.pkl")

# # # ===== MediaPipe =====
# # mp_hands = mp.solutions.hands
# # mp_draw = mp.solutions.drawing_utils
# # hands = mp_hands.Hands(static_image_mode=False,
# #                        max_num_hands=1,
# #                        min_detection_confidence=0.7,
# #                        min_tracking_confidence=0.7)

# # # ===== CustomTkinter Setup =====
# # ctk.set_appearance_mode("light")  # default light mode
# # ctk.set_default_color_theme("blue")

# # root = ctk.CTk()
# # root.title("Sign Language Detection")
# # root.geometry("1100x700")

# # # ===== Layout =====
# # # Left: Video & Subtitle
# # left_frame = ctk.CTkFrame(root, corner_radius=10)
# # left_frame.pack(side="left", fill="both", expand=True, padx=10, pady=10)

# # video_label = ctk.CTkLabel(left_frame, text="")
# # video_label.pack(pady=10)

# # subtitle_var = ctk.StringVar(value="Waiting...")
# # subtitle_label = ctk.CTkLabel(left_frame, textvariable=subtitle_var,
# #                               font=("Arial", 28, "bold"),
# #                               text_color="black")
# # subtitle_label.pack(pady=10)

# # # Right: Control Panel
# # right_frame = ctk.CTkFrame(root, width=500, corner_radius=10)
# # right_frame.pack(side="right", fill="y", padx=10, pady=10)

# # ctk.CTkLabel(right_frame, text="Settings", font=("Arial", 20, "bold")).pack(pady=20)

# # # Mode Toggle
# # def toggle_mode(choice):
# #     ctk.set_appearance_mode(choice.lower())

# # mode_option = ctk.CTkOptionMenu(right_frame,
# #                                 values=["Light", "Dark"],
# #                                 command=toggle_mode)
# # mode_option.set("Light")
# # mode_option.pack(pady=20)

# # # Subtitle Size
# # def change_size(choice):
# #     sizes = {"Small":20, "Medium":28, "Large":36}
# #     subtitle_label.configure(font=("Arial", sizes[choice], "bold"))

# # size_option = ctk.CTkOptionMenu(right_frame,
# #                                 values=["Small", "Medium", "Large"],
# #                                 command=change_size)
# # size_option.set("Medium")
# # size_option.pack(pady=20)

# # # Language Switch
# # language = ctk.StringVar(value="English")

# # def change_language(choice):
# #     language.set(choice)

# # lang_option = ctk.CTkOptionMenu(right_frame,
# #                                 values=["English", "Khmer"],
# #                                 command=change_language)
# # lang_option.set("English")
# # lang_option.pack(pady=20)

# # # ===== Webcam =====
# # cap = cv2.VideoCapture(0)

# # def extract_landmarks(hand_landmarks):
# #     coords = []
# #     for lm in hand_landmarks.landmark:
# #         coords.extend([lm.x, lm.y, lm.z])
# #     return coords

# # # Optional: simple English→Khmer mapping
# # khmer_dict = {
# #     "Again": "ម្តងទៀត",
# #     "Bathroom": "បង្គន់",
# #     "Eat": "បរិភោគ",
# #     "Find": "ស្វែងរក",
# #     "Fine": "ល្អ",
# #     "Good": "ល្អ",
# #     "Hello": "សួស្តី",
# #     "I_Love_You": "ខ្ញុំស្រឡាញ់អ្នក",
# #     "Like": "ចូលចិត្ត",
# #     "Me": "ខ្ញុំ",
# #     "Milk": "ទឹកដោះគោ",
# #     "No": "ទេ",
# #     "Please": "សូម",
# #     "See_You_Later": "ជួបគ្នាវិញ",
# #     "Sleep": "គេង",
# #     "Talk": "និយាយ",
# #     "Thank You": "អរគុណ",
# #     "Understand": "យល់",
# #     "Want": "ចង់",
# #     "What's Up": "មានអីថ្មី",
# #     "Who": "នរណា",
# #     "Why": "ហេតុអ្វី",
# #     "Yes": "បាទ/ចាស",
# #     "You": "អ្នក"
# # }

# # def update_frame():
# #     ret, frame = cap.read()
# #     if not ret:
# #         root.after(10, update_frame)
# #         return

# #     frame = cv2.flip(frame, 1)
# #     rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     result = hands.process(rgb_frame)

# #     prediction_text = "Waiting..."

# #     if result.multi_hand_landmarks:
# #         for handLms in result.multi_hand_landmarks:
# #             mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
# #             features = np.array(extract_landmarks(handLms)).reshape(1, -1)
# #             try:
# #                 pred = model.predict(features)[0]
# #                 if language.get() == "Khmer" and pred in khmer_dict:
# #                     prediction_text = khmer_dict[pred]
# #                 else:
# #                     prediction_text = pred
# #             except:
# #                 prediction_text = "Error"

# #     subtitle_var.set(prediction_text)

# #     # Display webcam feed
# #     img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
# #     img = Image.fromarray(img)
# #     imgtk = ImageTk.PhotoImage(image=img)
# #     video_label.imgtk = imgtk
# #     video_label.configure(image=imgtk)

# #     root.after(10, update_frame)

# # def on_close():
# #     cap.release()
# #     root.destroy()

# # root.protocol("WM_DELETE_WINDOW", on_close)
# # update_frame()
# # root.mainloop()



# import streamlit as st
# import cv2, mediapipe as mp, numpy as np, joblib

# st.title("Sign Language Detection")
# model = joblib.load("model/sign_rf_model.pkl")

# mp_hands = mp.solutions.hands
# mp_draw = mp.solutions.drawing_utils
# hands = mp_hands.Hands()

# run = st.checkbox("Start Camera")
# frame_window = st.image([])

# cap = None
# if run:
#     cap = cv2.VideoCapture(0)

# while run:
#     ret, frame = cap.read()
#     if not ret:
#         break
#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     res = hands.process(rgb)

#     if res.multi_hand_landmarks:
#         for hand in res.multi_hand_landmarks:
#             mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)
#             feats = []
#             for lm in hand.landmark:
#                 feats.extend([lm.x, lm.y, lm.z])
#             pred = model.predict([feats])[0]
#             cv2.putText(frame, pred, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,255,0),2)

#     frame_window.image(frame, channels="BGR")
import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import joblib
from PIL import Image

# ===== Load Model =====
model = joblib.load("model/sign_rf_model.pkl")

# ===== MediaPipe =====
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

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

# ===== Helper =====
def extract_landmarks(hand_landmarks):
    coords = []
    for lm in hand_landmarks.landmark:
        coords.extend([lm.x, lm.y, lm.z])
    return coords

# ===== Webcam =====
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Cannot access webcam.")
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    prediction_text = "Waiting..."

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)
            features = np.array(extract_landmarks(handLms)).reshape(1, -1)
            try:
                pred = model.predict(features)[0]
                if language == "Khmer" and pred in khmer_dict:
                    prediction_text = khmer_dict[pred]
                else:
                    prediction_text = pred
            except:
                prediction_text = "Error"

    frame_window.image(frame, channels="BGR")
    prediction_placeholder.markdown(
        f"<h2 style='text-align:center; font-size:{font_size}px'>{prediction_text}</h2>",
        unsafe_allow_html=True
    )
