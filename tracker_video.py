import cv2
import torch
import numpy as np
from ultralytics import YOLO
import sys
import os

# Add DeepSORT to path
sys.path.append("deepsort")
from deep_sort.deep_sort import DeepSort

# === Load Models ===
yolo_model = YOLO("yolov8x.pt")
deepsort = DeepSort(
    model_path="deepsort/deep_sort/deep/checkpoint/ckpt.t7",
    max_dist=0.1,
    min_confidence=0.3,
    nms_max_overlap=1.0,
    max_iou_distance=0.5,
    max_age=10,
    n_init=2,
    nn_budget=40,
    use_cuda=True
)

# === Input/Output Paths ===
video_path = "parsed_video.mp4"  # Input video made from detection frames
output_path = "output_tracked.mp4"

# === Read Video ===
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

# === Initialize Writer ===
out = cv2.VideoWriter(
    output_path,
    cv2.VideoWriter_fourcc(*"mp4v"),
    fps,
    (frame_width, frame_height)
)

frame_idx = 0
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # === YOLOv8 Inference ===
    results = yolo_model(frame, conf=0.1)  # Lower threshold to get more detections
    boxes = results[0].boxes

    detections = []
    classes = []

    for box in boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        if cls != 0:
            continue  # Only track 'person' class

        x1, y1, x2, y2 = map(int, box.xyxy[0])
        padding_ratio = 0.05
        w, h = x2 - x1, y2 - y1
        x1 -= int(w * padding_ratio)
        y1 -= int(h * padding_ratio)
        x2 += int(w * padding_ratio)
        y2 += int(h * padding_ratio)

        w, h = x2 - x1, y2 - y1
        cx, cy = x1 + w // 2, y1 + h // 2
        detections.append([cx, cy, w, h, conf])
        classes.append(cls)

    # === DeepSORT Tracking ===
    if detections:
        dets_np = np.array(detections)
        bbox_xywh = dets_np[:, :4]
        confidences = dets_np[:, 4]

        tracks, _ = deepsort.update(bbox_xywh, confidences, classes, frame)

        for track in tracks:
            if len(track) < 6:
                continue
            x1, y1, x2, y2, cls, track_id = map(int, track)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"ID {track_id}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    else:
        deepsort.increment_ages()

    out.write(frame)
    frame_idx += 1
    if frame_idx % 20 == 0:
        print(f"Processed {frame_idx} frames...")

cap.release()
out.release()
print("✅ Tracked video saved to:", output_path)