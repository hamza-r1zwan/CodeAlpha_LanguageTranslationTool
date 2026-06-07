# Object Detection and Tracking — CodeAlpha AI Internship

## Project Overview
A real-time object detection and tracking system built using YOLOv8 and DeepSORT.
Each detected object is assigned a unique tracking ID that persists across frames.

## Features
- Real-time object detection using YOLOv8 (pretrained on COCO dataset)
- Multi-object tracking with unique IDs using DeepSORT algorithm
- Color-coded bounding boxes per tracked object
- Processes video files and saves annotated output
- Live preview window during processing

## Technologies Used
- Python 3.11
- YOLOv8 (Ultralytics)
- DeepSORT (deep-sort-realtime)
- OpenCV (cv2)
- NumPy

## How to Run

### 1. Install dependencies
pip install ultralytics opencv-python deep-sort-realtime numpy

### 2. Add your video
Place your input video at: input/video.mp4

### 3. Run the script
py detect_track.py

### 4. View output
Output saved to: output/output_video.avi

## Project Structure
object_detection_tracking/
├── input/          # Place input video here
├── output/         # Annotated output video saved here
├── detect_track.py # Main script
└── README.md

## Internship
CodeAlpha AI Internship — Task 4: Object Detection and Tracking