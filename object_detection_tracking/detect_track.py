# ── Imports ──────────────────────────────────────────────────────────────────
import cv2
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# ── Load Models ──────────────────────────────────────────────────────────────
model = YOLO("yolov8n.pt")
tracker = DeepSort(max_age=30)

# ── Open Input Video ──────────────────────────────────────────────────────────
video_path = "input/video.mp4"
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("ERROR: Could not open video. Check the path.")
    exit()

# ── Get Video Properties ──────────────────────────────────────────────────────
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps    = int(cap.get(cv2.CAP_PROP_FPS))

# ── Setup Output Video Writer ─────────────────────────────────────────────────
output_path = "output/output_video.avi"
fourcc = cv2.VideoWriter_fourcc(*"XVID")
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# ── Color palette for different track IDs ────────────────────────────────────
COLORS = [
    (0, 255, 0),
    (255, 100, 0),
    (0, 100, 255),
    (255, 0, 255),
    (0, 255, 255),
]

# ── Main Processing Loop ──────────────────────────────────────────────────────
frame_count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    print(f"Processing frame {frame_count}...", end="\r")

    # Step A: Run YOLO detection
    results = model(frame, verbose=False)

    # Step B: Format detections for DeepSORT
    detections = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls_id = int(box.cls[0])
            cls_name = model.names[cls_id]

            if conf < 0.4:
                continue

            w = x2 - x1
            h = y2 - y1
            detections.append(([x1, y1, w, h], conf, cls_name))

    # Step C: Update tracker
    tracks = tracker.update_tracks(detections, frame=frame)

    # Step D: Draw boxes and labels
    for track in tracks:
        if not track.is_confirmed():
            continue

        track_id = int(track.track_id)          # FIX: convert to int
        cls_name = track.get_det_class()
        ltrb = track.to_ltrb()
        x1, y1, x2, y2 = map(int, ltrb)
        color = COLORS[track_id % len(COLORS)]

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        # Label background
        label = f"ID:{track_id} {cls_name}"
        (lw, lh), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(frame, (x1, y1 - lh - 8), (x1 + lw + 4, y1), color, -1)

        # Label text
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    # Step E: Write frame to output
    out.write(frame)

    # Live preview (press Q to quit early)
    cv2.imshow("Detection & Tracking", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ── Cleanup ───────────────────────────────────────────────────────────────────
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"\nDone! Output saved to: {output_path}")