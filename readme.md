# People Counting & Tracking with YOLOv11 + ByteTrack

## Overview

This project provides a **real-time people counting and tracking application** using [YOLOv11](https://github.com/ultralytics/ultralytics) for detection and [ByteTrack](https://github.com/ifzhang/ByteTrack) for multi-object tracking.  
A [Streamlit](https://streamlit.io/) dashboard displays live video with bounding boxes and unique IDs, plots occupancy over time, and logs results to a CSV file for further analysis.

---

## Features

- **Real-time people detection** using YOLOv11.
- **Multi-person tracking** with persistent IDs via ByteTrack.
- **Interactive Streamlit dashboard** with live video and occupancy plot.
- **CSV logging** of timestamp, people count, and all active IDs per frame.

---

## How It Works

1. **Detection:**  
   Each frame is processed by YOLOv11 to detect people.

2. **Tracking:**  
   Detected people are tracked with ByteTrack, which assigns and maintains unique IDs.

3. **Visualization:**  
   - Bounding boxes and IDs are drawn on the video.
   - The current people count is shown on the video.
   - Occupancy over time is plotted in real time.

4. **Logging:**  
   Each frame's timestamp, people count, and active IDs are saved to a CSV file.

---

## Usage

### 1. Install Dependencies

pip install ultralytics streamlit plotly opencv-python
text

### 2. Download the YOLOv11 Model

Place the `yolov11m.pt` model file in your working directory.

### 3. Prepare Your Video

- Set `video_path = "Video.mp4"` in the script, or use `0` for webcam.

### 4. Run the App

streamlit run your_script.py
text

### 5. View Results

- Open the Streamlit web interface (usually at [http://localhost:8501](http://localhost:8501)).
- Watch the live video with bounding boxes and IDs.
- View the occupancy chart.
- Check `people_tracking_bytetrack_log.csv` for the detection and tracking log.

---

## Output

- **people_tracking_bytetrack_log.csv**  
  Each row contains: `Timestamp, People_Count, Active_IDs`

---

## Requirements

- Python 3.8+
- [ultralytics](https://pypi.org/project/ultralytics/)
- [streamlit](https://pypi.org/project/streamlit/)
- [plotly](https://pypi.org/project/plotly/)
- [opencv-python](https://pypi.org/project/opencv-python/)

---

## Notes

- ByteTrack is used via Ultralytics' built-in tracking (`tracker="bytetrack.yaml"`).
- The code is easy to adapt for other object classes or video sources.
- For best performance, use a machine with a compatible GPU.

---
