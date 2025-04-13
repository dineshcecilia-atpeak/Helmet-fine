import cv2
import streamlit as st
import numpy as np
from ultralytics import YOLO
from paddleocr import PaddleOCR
from datetime import datetime
import pandas as pd
from io import StringIO
from PIL import Image
import io

# Initialize model and OCR
model = YOLO("best.pt")  # Replace with your YOLOv8 trained model path
ocr = PaddleOCR()
names = model.names

# Define polygon area (adjust as per your frame)
area = [(1, 173), (62, 468), (608, 431), (364, 155)]

# Store fine log
if "fine_log" not in st.session_state:
    st.session_state.fine_log = []

# Title and video source selection
st.title("🚨 Helmet Violation Detection")
video_source = st.selectbox("Select Video Source", ["Webcam", "Video File"])
video_file = None
if video_source == "Video File":
    video_file = st.file_uploader("Upload video", type=["mp4", "avi"])

start = st.button("Start Detection", key="start_btn")

# Sidebar - Violation History
with st.sidebar:
    st.header("📝 Violation History")
    if st.session_state.fine_log:
        for idx, record in enumerate(st.session_state.fine_log[::-1]):
            st.markdown(f"**{record['Number Plate']}**  \n📅 {record['Date']} ⏰ {record['Time']}")

# Main Detection Process
if start:
    if video_source == "Webcam":
        cap = cv2.VideoCapture(0)
    else:
        if video_file is not None:
            tfile = open("temp_video.mp4", "wb")
            tfile.write(video_file.read())
            cap = cv2.VideoCapture("temp_video.mp4")
        else:
            st.warning("Upload a video to proceed.")
            st.stop()

    stframe = st.empty()
    processed_ids = set()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (1020, 500))
        results = model.track(frame, persist=True)

        no_helmet_detected = False
        numberplate_box = None
        numberplate_id = None

        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.int().cpu().tolist()
            class_ids = results[0].boxes.cls.int().cpu().tolist()
            track_ids = results[0].boxes.id.int().cpu().tolist()

            for box, cls, tid in zip(boxes, class_ids, track_ids):
                c = names[cls]
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                if cv2.pointPolygonTest(np.array(area, np.int32), (cx, cy), False) >= 0:
                    if c == 'no-helmet':
                        no_helmet_detected = True
                    elif c == 'numberplate':
                        numberplate_box = box
                        numberplate_id = tid

            if no_helmet_detected and numberplate_box and numberplate_id not in processed_ids:
                x1, y1, x2, y2 = numberplate_box
                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (120, 85))
                text = ocr.ocr(crop, rec=True)
                detected = ''.join([res[1][0] for res in text[0]]) if text[0] else "Unknown"
                current_time = datetime.now().strftime('%H:%M:%S')
                current_date = datetime.now().strftime('%Y-%m-%d')

                # Convert crop to image and store as a reference for table
                pil_img = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
                img_byte_arr = io.BytesIO()
                pil_img.save(img_byte_arr, format='PNG')
                img_byte_arr = img_byte_arr.getvalue()

                # Storing the fine log with image and text
                st.session_state.fine_log.append({
                    "Number Plate": detected,
                    "Date": current_date,
                    "Time": current_time,
                    "Image": img_byte_arr  # Keep the image in byte format for display
                })
                processed_ids.add(numberplate_id)

        # Draw polygon and show video
        cv2.polylines(frame, [np.array(area, np.int32)], True, (255, 0, 255), 2)
        stframe.image(frame, channels="BGR")

    cap.release()

# Show fine log as table
if st.session_state.fine_log:
    st.subheader("🚔 Fined Vehicles Log")
    df = pd.DataFrame(st.session_state.fine_log)

    # Display DataFrame excluding image column
    st.table(df.drop(columns=["Image"]))  # Drop Image column from table (images will be displayed separately)

    # CSV download button
    csv = df.drop(columns=["Image"]).to_csv(index=False)  # Remove image column from CSV export
    st.download_button(
        label="Download Fine Log as CSV",
        data=csv,
        file_name="fine_log.csv",
        mime="text/csv"
    )

    # Display snapshots of captured number plates
    st.subheader("📸 Captured Number Plate Images")
    for record in st.session_state.fine_log:
        st.image(record["Image"], caption=f"Captured Number Plate: {record['Number Plate']}", use_container_width=True)
