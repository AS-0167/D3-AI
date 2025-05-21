import streamlit as st
import cv2
import os
import tempfile
import time
from cnn import CNN
from PIL import Image

# Instantiate the model (ensure model is loaded properly inside CNN class)
model_path = "./d3_cnn_model.ckpt"
model = CNN()
model.load_model(model_path)

# Constants
FRAME_INTERVAL = 5  # seconds

# Function to extract frames every 5 seconds
def extract_frames(video_path, interval=5):
    frames = []
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        st.error("Error: Cannot open video file.")
        return []

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    success = True
    frame_idx = 0
    while success:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        success, frame = cap.read()
        if success:
            tmp_img_path = os.path.join(tempfile.gettempdir(), f"frame_{frame_idx}.jpg")
            cv2.imwrite(tmp_img_path, frame)
            frames.append(tmp_img_path)
            frame_idx += frame_interval
        if frame_idx >= frame_count:
            break

    cap.release()
    return frames

# Streamlit UI
st.set_page_config(page_title="Driver Image/Video Prediction", layout="centered")
st.title("Driver Behavior Prediction")

uploaded_file = st.file_uploader("Upload an image or video", type=["jpg", "jpeg", "png", "mp4", "avi", "mov"])

if uploaded_file is not None:
    with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(uploaded_file.name)[1]) as tmp_file:
        tmp_file.write(uploaded_file.read())
        file_path = tmp_file.name

    if uploaded_file.type.startswith("image"):
        st.image(file_path, caption="Uploaded Image", use_container_width=True)
        prediction = model.predict(file_path, show_image=False)
        if prediction:
            pred_class, confidence = prediction
            st.success(f"Prediction: **{pred_class}** (Confidence: {confidence:.2f})")
        else:
            st.error("Prediction failed.")

    elif uploaded_file.type.startswith("video"):
        st.video(file_path)
        st.info("Extracting frames every 5 seconds...")
        frames = extract_frames(file_path, interval=FRAME_INTERVAL)

        if frames:
            st.success(f"{len(frames)} frames extracted.")
            for idx, frame_path in enumerate(frames):
                st.image(frame_path, caption=f"Frame {idx+1}", use_container_width=True)
                prediction = model.predict(frame_path, show_image=False)
                if prediction:
                    pred_class, confidence = prediction
                    st.write(f"Prediction: **{pred_class}** (Confidence: {confidence:.2f})")
                else:
                    st.write("Prediction failed.")
                time.sleep(1)  # Delay for effect
        else:
            st.warning("No frames extracted from video.")

    # Button to reset (simulate next upload)
    if st.button("Upload Another Image/Video"):
        st.rerun()
