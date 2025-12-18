# Traffic Controller – YOLO + OpenCV + Adaptive Signal Control

## Overview

This script implements an **end-to-end intelligent traffic control system** using a live camera feed, YOLOv8 vehicle detection, centroid tracking, lane-based queue estimation, and adaptive traffic signal control via a NodeMCU device.

The system supports:

- Manual **lane calibration** using a captured camera frame
- **Vehicle detection** using YOLOv8
- **Persistent vehicle tracking** with unique IDs
- **Queue length estimation** in meters
- **Adaptive green time allocation**
- **Emergency vehicle prioritization**
- **Non-blocking traffic signal control** via HTTP

---

## System Architecture

```
Camera → YOLOv8 Detection → Centroid Tracker
        → Lane Assignment → Queue Estimation
        → Adaptive Controller → NodeMCU (Traffic Lights)
```

---

## Features

### 1. Lane Calibration (GUI)

- Capture a frame from the camera
- Draw polygon regions for each lane
- Optional homography calibration to map pixels → real-world meters
- Lane data stored in `lanes_config.json`

### 2. Vehicle Detection

- Uses **YOLOv8 (Ultralytics)**
- Detects:

  - car
  - bus
  - truck
  - motorbike
  - ambulance
  - fire engine / police (emergency)

### 3. Vehicle Tracking

- Lightweight **centroid tracker**
- Assigns persistent IDs to vehicles
- Avoids double counting

### 4. Queue Length Estimation

- Counts unique vehicles per lane
- Converts count → queue length in meters
- Uses average vehicle length (default: `4.5m`)
- Optional homography improves accuracy

### 5. Adaptive Traffic Control

- Selects lane with highest queue length
- Dynamically assigns green time
- Enforces:

  - Minimum green time
  - Maximum green time

- Sends commands to NodeMCU via HTTP

### 6. Emergency Vehicle Priority

- Detects emergency vehicles
- Immediately switches signal to green for that lane
- Overrides normal adaptive logic

---

## Project Files

```
traffic-controller/
│
├── traffic_controller.py     # Main application
├── lanes_config.json         # Saved lane polygons & homography
├── calib_frame.jpg           # Captured calibration image
└── requirements.txt
```

---

## Installation

### 1. Create Virtual Environment

```bash
cd traffic-controller
python3 -m venv venv
```

### 2. Activate Virtual Environment

**macOS / Linux**

```bash
source venv/bin/activate
```

**Windows**

```bash
venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install ultralytics opencv-python numpy requests
```

---

## Configuration

Edit the following values inside `traffic_controller.py` if required:

```python
NODEMCU_IP = "192.168.1.7"
YOLO_WEIGHTS = "yolov8n.pt"
CONF = 0.45
MIN_GREEN_MS = 3000
MAX_GREEN_MS = 10000
```

---

## Usage

Run the program:

```bash
python traffic_controller.py
```

You will be prompted to choose a mode.

---

## Mode 1: Lane Calibration (`c`)

### Steps

1. Select **`c`** when prompted
2. Live camera feed opens
3. Press **`c`** to capture a frame
4. Draw lanes:

   - **Left click** → add point
   - **Right click** → finish lane polygon

5. Press:

   - **`s`** → save lanes
   - **`r`** → reset
   - **`q`** → quit

### Optional: Homography Calibration

- Press **`h`**
- Click **4 image points** (clockwise)
- Enter real-world meter coordinates
- Enables pixel → meter conversion

---

## Mode 2: Live Detection (`w`)

### Functionality

- Starts webcam detection
- Loads lane polygons from `lanes_config.json`
- Displays:

  - Vehicle bounding boxes
  - Vehicle IDs
  - Lane queue length (meters)

- Controls traffic signals in real time

### Controls

- **`q`** → quit
- **`p`** → save snapshot

---

## Adaptive Signal Logic

### Green Time Calculation

```text
queue_meters → estimated_vehicles → required_time → clipped to min/max
```

- Uses conservative saturation flow rate
- Prevents rapid signal switching
- Falls back to all-red if no vehicles detected

---

## Emergency Handling

If an emergency vehicle is detected:

- Signal switches immediately to green
- All other lanes set to red
- Remains green for a fixed emergency duration

---

## NodeMCU Communication

Signals are controlled via HTTP requests:

```
GET /control?lane=Lane_1&color=green&state=1
GET /mode?set=manual
```

Requests are:

- Non-blocking
- Retried automatically on failure

---

## Limitations

- Queue length estimation is approximate
- Homography improves accuracy but is optional
- Best suited for:

  - Single intersection
  - Fixed camera setup
  - Controlled environments (toy or real traffic)

---

## Future Improvements

- Train custom YOLO model for toy vehicles
- Multi-camera intersection support
- Speed and density estimation
- DeepSORT / ByteTrack integration
- Cloud-based monitoring dashboard

---

## Conclusion

This system demonstrates a **complete intelligent traffic signal controller** using computer vision and adaptive control logic. It is suitable for academic projects, simulations, and prototype smart traffic systems.