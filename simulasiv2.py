import numpy as np
import random
import streamlit as st
import streamlit_webrtc as webrtc
import tensorflow as tf
import mediapipe as mp
from PIL import Image

# Load the TFLite model for ASL letter detection
interpreter = tf.lite.Interpreter(model_path="model_asl.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7)

# Function to process the image and get hand landmarks
def process_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    if results.multi_hand_landmarks:
        landmarks = []
        for hand_landmarks in results.multi_hand_landmarks:
            for landmark in hand_landmarks.landmark:
                landmarks.append((landmark.x, landmark.y, landmark.z))
        return np.array(landmarks).flatten()
    else:
        return np.zeros(63)  # Return a zero array if no hands are detected

# Function to predict ASL using the TFLite model
def predict_asl(data):
    data = np.array(data, dtype=np.float32).reshape(1, 63)
    interpreter.set_tensor(input_details[0]["index"], data)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]["index"])
    return np.argmax(prediction)

# Streamlit sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["ASL Game & Simulation", "About ASL", "Creators"])

# Halaman Utama: Mini-Game ASL dan Simulasi
if page == "ASL Game & Simulation":
    st.title("ASL Practice & Detection")
    st.write("Select an option to start:")
    
    mode = st.radio("Choose Mode:", ["Mini-Game", "ASL Simulation"])
    
    # Webcam Placeholder
    if mode == "Mini-Game":
        st.subheader("Mini-Game: Match the ASL letter")
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        current_letter = random.choice(letters)
        score = 0
        
        # Display target letter and score
        target_letter = st.empty()
        score_display = st.empty()
        target_letter.write(f"Target Letter: **{current_letter}**")
        
        def video_frame_callback(frame):
            img = frame.to_ndarray(format="bgr24")
            landmarks = process_image(img)
            predicted_label = predict_asl(landmarks)
            detected_letter = letters[predicted_label]
            
            if detected_letter == current_letter:
                score += 1
                current_letter = random.choice(letters)
                target_letter.write(f"Target Letter: **{current_letter}**")
            
            return frame
        
        webrtc_streamer(key="example", video_frame_callback=video_frame_callback)

    # Mode ASL Simulation
    elif mode == "ASL Simulation":
        st.write("This mode simulates ASL detection.")

# Halaman Tentang ASL
elif page == "About ASL":
    st.title("About American Sign Language (ASL)")
    st.write("American Sign Language (ASL) is a visual language...")

# Halaman Pembuat
elif page == "Creators":
    st.title("Creators")
    st.write("This application was created by a team of dedicated developers.")
