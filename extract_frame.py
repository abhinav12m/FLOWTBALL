import cv2
import os

video_path = "Input.mp4"
output_dir = "frames"
fps_target = 15

# Create the output folder
os.makedirs(output_dir, exist_ok=True)

# Read video
cap = cv2.VideoCapture(video_path)
original_fps = cap.get(cv2.CAP_PROP_FPS)
interval = int(round(original_fps / fps_target))

frame_idx = 0
saved_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    if frame_idx % interval == 0:
        filename = os.path.join(output_dir, f"frame_{saved_idx:04d}.jpg")
        cv2.imwrite(filename, frame)
        saved_idx += 1

    frame_idx += 1

cap.release()
print(f"Done! Saved {saved_idx} frames in the folder: {output_dir}")