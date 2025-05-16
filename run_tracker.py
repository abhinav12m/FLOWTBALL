import os
import cv2
import sys
import numpy as np
from ultralytics import YOLO

# Add DeepSORT repo to import path
sys.path.append("deepsort")
from deep_sort.deep_sort import DeepSort

# Load YOLOv8 model and DeepSORT tracker
yolo = YOLO("yolov8x.pt")
deepsort = DeepSort("deepsort/deep_sort/deep/checkpoint/ckpt.t7")

# Paths
input_folder = "frames"
output_folder = "tracked_frames"
os.makedirs(output_folder, exist_ok=True)

# Get all frame filenames
frame_files = sorted(f for f in os.listdir(input_folder) if f.endswith(".jpg"))

for idx, fname in enumerate(frame_files):
    if idx < 130 or idx > 400:
        continue

    frame_path = os.path.join(input_folder, fname)
    frame = cv2.imread(frame_path)

    # Run YOLOv8 detection
    results = yolo(frame, conf=0.25)
    boxes = results[0].boxes

    # Collect detections: [x, y, w, h, conf]
    detections = []
    for box in boxes:
        if int(box.cls[0]) != 0:  # keep only person class
            continue
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        w, h = x2 - x1, y2 - y1
        conf = float(box.conf[0])
        detections.append([x1, y1, w, h, conf])

    if detections:
        dets_np = np.array(detections)
        bboxes = dets_np[:, :4]
        confidences = dets_np[:, 4]
        classes = np.zeros(len(dets_np))  # class 0 = person

        # Update DeepSORT tracker
        tracks_output = deepsort.update(bboxes, confidences, classes, frame)
        tracks = tracks_output[0]

        for track in tracks:
            x1, y1, x2, y2, _, track_id = track
            x1, y1, x2, y2, track_id = map(int, [x1, y1, x2, y2, track_id])

            # Draw bounding box and ID
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        deepsort.increment_ages()

    # Save tracked frame
    cv2.imwrite(os.path.join(output_folder, fname), frame)

    if idx % 25 == 0:
        print(f"Processed frame {idx + 1}/{len(frame_files)}")

print("✅ Tracking completed for frames 130-400. Output saved to tracked_frames/")