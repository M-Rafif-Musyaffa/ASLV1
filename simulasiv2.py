from PIL import Image
import mediapipe as mp
import numpy as np
import random
import time
import streamlit as st
import tensorflow as tf

# Load the TFLite model for ASL letter detection
interpreter = tf.lite.Interpreter(model_path="model_asl.tflite")
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Set up MediaPipe for hand detection
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True, max_num_hands=1, min_detection_confidence=0.7
)

# Streamlit sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["ASL Game & Simulation", "About ASL", "Creators"])

# Halaman Utama: Mini-Game ASL dan Simulasi
if page == "ASL Game & Simulation":
    st.title("ASL Practice & Detection")
    st.write("Select an option to start:")

    # Tombol untuk memilih mode
    mode = st.radio("Choose Mode:", ["Mini-Game", "ASL Simulation"])

    # Webcam placeholder
    frame_placeholder = st.empty()
    target_letter = st.empty()
    score_text = st.empty()

    # Inisialisasi kamera
    cap = cv2.VideoCapture(0)  # Anda mungkin tetap memerlukan OpenCV di sini

    # Fungsi untuk memproses gambar
    def image_processed(hand_img):
        img_rgb = cv2.cvtColor(hand_img, cv2.COLOR_BGR2RGB)
        img_flip = cv2.flip(img_rgb, 1)
        output = hands.process(img_flip)
        try:
            data = output.multi_hand_landmarks[0]
            data = str(data).strip().split("\n")
            garbage = ["landmark {", "  visibility: 0.0", "  presence: 0.0", "}"]
            clean = [float(i.strip()[2:]) for i in data if i not in garbage]
            return clean
        except:
            return np.zeros([1, 63], dtype=int)[0]

    # Fungsi untuk prediksi menggunakan model TFLite
    def predict_asl_tflite(data):
        data = np.array(data, dtype=np.float32).reshape(1, 63)
        interpreter.set_tensor(input_details[0]["index"], data)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]["index"])
        predicted_label = np.argmax(prediction)
        return predicted_label

    # Mode Mini-Game
    if mode == "Mini-Game":
        st.subheader("Mini-Game: Match the ASL letter")
        letters = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        current_letter = random.choice(letters)
        score = 0

        # Display target letter and score
        target_letter.write(f"Target Letter: **{current_letter}**")

        correct_display_time = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                st.write("Unable to access the camera.")
                break

            data = image_processed(frame)
            y_pred = predict_asl_tflite(data)

            detected_letter = letters[y_pred]

            if detected_letter == current_letter:
                score += 1
                score_text.write(f"**Score**: {score}")
                current_letter = random.choice(letters)
                target_letter.write(f"Target Letter: **{current_letter}**")
                correct_display_time = time.time()

            if time.time() - correct_display_time < 1:
                cv2.putText(
                    frame,
                    "Correct!",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    2,
                    (0, 255, 0),
                    3,
                )

            # Tampilkan webcam feed di Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()

# Halaman Tentang ASL
elif page == "About ASL":
    st.title("About American Sign Language (ASL)")
    st.write("American Sign Language (ASL) is a visual language...")

# Halaman Pembuat
elif page == "Creators":
    st.title("Creators")
    st.write("This application was created by a team of dedicated developers.")
    # Detail pencipta...
