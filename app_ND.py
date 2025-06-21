import cv2
from ultralytics import YOLO
from datetime import datetime
import csv
import os
import time
import json
import paho.mqtt.client as mqtt

# ----------------------------
# CONFIGURATION
# ----------------------------
video_path = "Video.mp4"               # Input video file
model_path = "yolo11m.pt"              # Use yolo11m or any other YOLOv8 model
mqtt_broker = "broker.hivemq.com"       # Replace with your broker
mqtt_port = 1883
mqtt_topic = "people_counter/occupancy"
mqtt_interval = 10                      # seconds between publishes
csv_filename = "people_tracking_log.csv"

# ----------------------------
# MQTT Setup
# ----------------------------
mqtt_client = mqtt.Client()
mqtt_client.connect(mqtt_broker, mqtt_port, 60)
last_mqtt_publish = time.time()

# ----------------------------
# Load Model & Setup Video
# ----------------------------
model = YOLO(model_path)
cap = cv2.VideoCapture(video_path)

# Setup CSV Logging
csv_exists = os.path.isfile(csv_filename)
with open(csv_filename, mode='a', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    if not csv_exists:
        csvwriter.writerow(['Timestamp', 'Occupancy'])

    # ----------------------------
    # Main Loop
    # ----------------------------
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame,
            classes=[0],  # Only detect people
            tracker="bytetrack.yaml",
            persist=True
        )

        current_ids = set()

        if hasattr(results[0], "boxes") and results[0].boxes is not None:
            for det in results[0].boxes:
                if hasattr(det, "id") and det.id is not None:
                    tid = int(det.id.cpu().numpy()[0])
                    current_ids.add(tid)

        # Occupancy = number of unique tracked people in frame
        occupancy = len(current_ids)
        timestamp = datetime.now().isoformat()

        # Log to CSV
        csvwriter.writerow([timestamp, occupancy])

        # Publish to MQTT if interval met
        current_time = time.time()
        if current_time - last_mqtt_publish >= mqtt_interval:
            payload = json.dumps({
                "timestamp": timestamp,
                "occupancy": occupancy
            })
            mqtt_client.publish(mqtt_topic, payload)
            last_mqtt_publish = current_time

        # Optional: display live frame with people count (can be removed in headless mode)
        cv2.putText(
            frame,
            f"PEOPLE: {occupancy}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2
        )
        cv2.imshow("People Tracker", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Cleanup
cap.release()
cv2.destroyAllWindows()
