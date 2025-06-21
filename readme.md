# People Counting & Tracking with YOLOv11 + ByteTrack

## Overview

This project provides a **real-time people counting and tracking application** using [YOLOv11](https://github.com/ultralytics/ultralytics) for detection and [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking.  
It logs occupancy data (number of people in frame) with timestamps and publishes this information to an MQTT broker at fixed intervals.

---

## Features

- **Real-time people detection** using YOLOv11.
- **Multi-person tracking** with persistent IDs via ByteTrack.
- **CSV logging** of timestamp and people count per frame.
- **MQTT publishing** of occupancy data every fixed interval (e.g. 10 seconds).
- Designed for **headless edge deployment**, with optional video display.

---

## How It Works

1. **Detection:**  
   Each frame is processed by YOLOv11 to detect people (`class=0`).

2. **Tracking:**  
   ByteTrack assigns persistent IDs to detected individuals to track them across frames.

3. **Counting:**  
   Unique tracked IDs in each frame are counted to determine room occupancy.

4. **Logging & Publishing:**  
   - Each frame's timestamp and occupancy are logged to CSV.
   - At regular intervals, occupancy is published to an MQTT topic.

---

## Usage

### 1. Install Dependencies

```bash
pip install ultralytics opencv-python paho-mqtt
```

### 2. Download the YOLOv11 Model

Place the `yolo11m.pt` (or other YOLOv11 model) in your working directory. You can get it from:

```bash
from ultralytics import YOLO
YOLO('yolov8n.pt')
```

### 3. Configure Script

- Set `video_path = "Video.mp4"` or `0` for webcam.
- Set your MQTT broker URL and topic.

### 4. Run the Script

```bash
python app_ND.py
```

### 5. Monitor Output

- Check `people_tracking_log.csv` for logs.
- Subscribe to your MQTT topic to view published occupancy data.

---

## Output

- **people_tracking_log.csv**  
  Each row contains: `Timestamp, Occupancy`

- **MQTT Payload Example:**
```json
{
  "timestamp": "2025-06-17T14:27:10.512789",
  "occupancy": 12
}
```

---

## Requirements

- Python 3.8+
- [ultralytics](https://pypi.org/project/ultralytics/)
- [opencv-python](https://pypi.org/project/opencv-python/)
- [paho-mqtt](https://pypi.org/project/paho-mqtt/)

---

## Notes

- ByteTrack is used via Ultralytics' built-in tracking (`tracker="bytetrack.yaml"`).
- Code can be extended to support region-based counting, MQTT authentication, or camera stream inputs.
- GPU recommended for real-time performance.

---
