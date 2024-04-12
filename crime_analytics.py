import streamlit as st
import cv2
import dlib
import numpy as np
import os
from scipy.spatial import distance

# Initialize dlib's face detector and facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

def extract_eyebrow_eye_features(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    rects = detector(gray, 1)

    if len(rects) > 0:
        rect = rects[0]  # Use the first detected face
        landmarks = predictor(gray, rect)

        # Extract the coordinates of the eyebrows and eyes landmarks
        points = []
        for i in range(36, 48):  # Eyes landmarks
            part = landmarks.part(i)
            points.append((part.x, part.y))
        for i in range(17, 27):  # Eyebrows landmarks
            part = landmarks.part(i)
            points.append((part.x, part.y))
        return points
    return None

def compare_landmarks(landmarks1, landmarks2):
    if landmarks1 is None or landmarks2 is None:
        return 0
    assert len(landmarks1) == len(landmarks2), "Landmark points do not match."
    distances = [distance.euclidean(landmarks1[i], landmarks2[i]) for i in range(len(landmarks1))]
    score = 1 / (1 + np.mean(distances))  # Simple scoring to inverse the mean distance
    return score

# Set the page configuration
st.set_page_config(
    # layout="wide",
    page_title='Identity Quest',
    # page_icon=img
)

st.markdown("<h1 style='text-align: center; color: #12ABDB; pb:4'>Identity Quest</h1>", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    bytes_data = uploaded_file.read()
    uploaded_image = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    # st.write("Uploaded Image:")
    # st.image(uploaded_image, channels="BGR")  # Display the uploaded image
    uploaded_image_landmarks = extract_eyebrow_eye_features(uploaded_image)

    if uploaded_image_landmarks:
        database_dir = "./database_named/unmasked"
        highest_score = 0
        best_match = None
        best_match_filename = ""

        for filename in os.listdir(database_dir):
            path = os.path.join(database_dir, filename)
            image = cv2.imread(path)
            image_landmarks = extract_eyebrow_eye_features(image)

            score = compare_landmarks(uploaded_image_landmarks, image_landmarks)
            if score > highest_score:
                highest_score = score
                best_match = image
                best_match_filename = filename

        if best_match is not None:
            st.success(f"Best match: {best_match_filename} with a score of {highest_score}")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.write("Uploaded Image:")
                st.image(uploaded_image, channels="BGR")
            with col2:
                st.write(f"{best_match_filename}")
                st.image(best_match, channels="BGR")
        else:
            st.info("No matching image found.")
    else:
        st.info("Could not detect facial landmarks in the uploaded image.")
