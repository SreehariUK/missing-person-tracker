import cv2
import face_recognition
import numpy as np
import streamlit as st
import tempfile
from datetime import datetime
import time

# CONFIGURATION
CAMERA_STREAMS = {
    "Camera 1": {
        "url": "http://192.168.137.198:8080/video",
        "location": (12.9716, 77.5946)
    },
    "Camera 2": {
        "url": "http://192.168.137.220:8080/video",
        "location": (12.2958, 76.6394)
    }
}

# STREAMLIT UI
st.title("🎯 Missing Person Real-Time Tracker - Scanner Mode")
st.markdown("Upload an image of the missing person. Once you start scanning, the system will check all cameras continuously.")

uploaded_file = st.file_uploader("📤 Upload Image", type=["jpg", "jpeg", "png"])
start_scanning = st.checkbox("🔍 Start Scanning")

if uploaded_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(uploaded_file.read())
        image_path = tmp_file.name

    # Load and encode uploaded image
    image = face_recognition.load_image_file(image_path)
    try:
        target_encoding = face_recognition.face_encodings(image)[0]
        st.success("✅ Face image uploaded and processed.")
    except IndexError:
        st.error("❌ No face found in uploaded image. Please try another.")
        st.stop()

    match_placeholder = st.empty()

    if start_scanning:
        scanning = True
        while scanning:
            for cam_name, cam_data in CAMERA_STREAMS.items():
                match_placeholder.info(f"🌀 Scanning {cam_name}...")

                cap = cv2.VideoCapture(cam_data['url'])
                ret, frame = cap.read()
                cap.release()

                if not ret or frame is None:
                    match_placeholder.error(f"❌ Failed to read from {cam_name}")
                    continue

                # Resize and convert frame
                small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

                face_locations = face_recognition.face_locations(rgb_small_frame)
                if face_locations:
                    try:
                        face_encodings = face_recognition.face_encodings(rgb_small_frame, known_face_locations=face_locations)
                    except Exception as e:
                        match_placeholder.error(f"⚠️ Error in face encoding: {e}")
                        continue

                    for face_encoding in face_encodings:
                        matches = face_recognition.compare_faces([target_encoding], face_encoding, tolerance=0.5)
                        if True in matches:
                            detected_time = datetime.now().strftime('%H:%M:%S')
                            lat, lon = cam_data['location']
                            match_placeholder.success(
                                f"✅ Match found in **{cam_name}** at **{detected_time}**.\n\n📍 Location: Latitude {lat}, Longitude {lon}"
                            )
                            st.balloons()
                            scanning = False  # Stop scanning flag
                            break  # Break inner for-loop (face_encodings)

                if not scanning:
                    break  # Break outer for-loop (cameras)

            if not scanning:
                break  # Break while-loop
            time.sleep(2)
