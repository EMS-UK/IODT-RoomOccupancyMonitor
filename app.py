import cv2
import streamlit as st
import plotly.graph_objs as go
from ultralytics import YOLO
from datetime import datetime
from collections import deque
import csv
import os

# -----------------------------------------------------------------------------
# People Counting & Tracking with YOLOv11 + ByteTrack
# -----------------------------------------------------------------------------
# This script detects and tracks people in a video stream using YOLOv11 and
# ByteTrack (via Ultralytics' built-in tracking). It displays live results in
# a Streamlit dashboard and logs occupancy data to a CSV file.
# -----------------------------------------------------------------------------

# Initialize the YOLOv11 model (for person detection only)
model = YOLO('yolo11m.pt')  # Ensure model file is present

# Streamlit dashboard setup
st.set_page_config(layout="wide")
st.title("People Counting & Tracking with YOLOv11 + ByteTrack")

# Set video source: use 0 for webcam or provide a video file path
video_path = "Video.mp4"
cap = cv2.VideoCapture(video_path)

# Streamlit UI placeholders for video and plot
frame_window = st.image([])
plot_area = st.empty()

# Deques for tracking occupancy history and timestamps (for plotting)
occupancy_history = deque(maxlen=100)
timestamps = deque(maxlen=100)

# CSV logging setup: create or append to a CSV file for tracking data
csv_filename = "people_tracking_bytetrack_log.csv"
csv_exists = os.path.isfile(csv_filename)
with open(csv_filename, mode='a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if not csv_exists:
        # Write CSV header if file is new
        csvwriter.writerow(['Timestamp', 'People_Count', 'Active_IDs'])

    # Main video processing loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            # Break the loop if video ends or cannot be read
            break

        # Run YOLOv11 detection and ByteTrack tracking on the current frame.
        # 'classes=[0]' restricts detection to 'person' class only.
        # 'tracker="bytetrack.yaml"' uses the built-in ByteTrack configuration.
        results = model.track(
            frame,
            classes=[0],
            tracker="bytetrack.yaml",
            persist=True
        )

        # Set to store unique IDs of tracked people in the current frame
        current_ids = set()

        # Check if detections are present and extract bounding boxes and IDs
        if hasattr(results[0], "boxes") and results[0].boxes is not None:
            for det in results[0].boxes:
                # Only process detections with valid IDs (i.e., tracked objects)
                if hasattr(det, "id") and det.id is not None:
                    tid = int(det.id.cpu().numpy()[0])
                    current_ids.add(tid)
                    x1, y1, x2, y2 = map(int, det.xyxy.cpu().numpy()[0])
                    # Draw bounding box and ID label on the frame
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(
                        frame,
                        f"ID: {tid}",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

        # Count the number of people currently tracked in the frame
        people_count = len(current_ids)

        # Overlay the current people count on the frame
        cv2.putText(
            frame,
            f"PEOPLE: {people_count}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )

        # Display the processed frame in the Streamlit app
        frame_window.image(frame, channels="BGR")

        # Update occupancy history and timestamps for trend plotting
        current_time = datetime.now().strftime('%H:%M:%S')
        timestamps.append(current_time)
        occupancy_history.append(people_count)

        # Log the current frame's data to the CSV file
        csvwriter.writerow([current_time, people_count, list(current_ids)])

        # Create and display a Plotly line chart of occupancy over time
        fig = go.Figure([
            go.Scatter(
                x=list(timestamps),
                y=list(occupancy_history),
                mode='lines+markers'
            )
        ])
        fig.update_layout(
            title='Room Occupancy Over Time',
            xaxis_title='Time',
            yaxis_title='People Count',
            yaxis=dict(range=[0, max(occupancy_history) + 5])
        )
        plot_area.plotly_chart(fig, use_container_width=True)

# Release the video capture object when done
cap.release()
