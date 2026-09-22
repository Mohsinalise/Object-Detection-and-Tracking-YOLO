import streamlit as st
import cv2
import tempfile
from ultralytics import YOLO
import time

st.set_page_config(page_title="YOLO Object Tracking", page_icon="🎯", layout="wide")

st.title("🎯 Real-Time Object Detection & Tracking (YOLOv8)")
st.write("Upload a video to perform object detection and persistent tracking.")

st.sidebar.header("Settings")
conf_val = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.4, 0.05)
tracker_type = st.sidebar.selectbox("Tracker", ["bytetrack.yaml", "botsort.yaml"])

@st.cache_resource
def load_model():
    return YOLO("yolov8n.pt")

model = load_model()

uploaded_file = st.sidebar.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    
    cap = cv2.VideoCapture(tfile.name)
    st_frame = st.empty()
    stop_btn = st.button("Stop")

    prev_time = time.time()
    while cap.isOpened() and not stop_btn:
        ret, frame = cap.read()
        if not ret:
            st.info("Video processing finished.")
            break

        results = model.track(frame, conf=conf_val, tracker=tracker_type, persist=True, verbose=False)
        annotated_frame = results[0].plot()

        now = time.time()
        fps = 1.0 / (now - prev_time) if now != prev_time else 0.0
        prev_time = now

        annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        st_frame.image(annotated_frame_rgb, channels="RGB", caption=f"FPS: {fps:.1f}", use_container_width=True)

    cap.release()
else:
    st.info("👈 Please upload a video file from the sidebar to start.")
