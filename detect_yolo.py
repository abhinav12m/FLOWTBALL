from ultralytics import YOLO
import os
from PIL import Image

# Load YOLOv8 model
model = YOLO("yolov8x.pt")

# Input/output paths
input_folder = "frames"
output_folder = "detections"
os.makedirs(output_folder, exist_ok=True)

# Inference on all frames
image_files = sorted([f for f in os.listdir(input_folder) if f.endswith(".jpg")])

for i, img_file in enumerate(image_files):
    img_path = os.path.join(input_folder, img_file)
    results = model(img_path, save=False, conf=0.25, iou=0.5, max_det=40)

    # Save the result with boxes drawn
    result_img = results[0].plot()
    out_path = os.path.join(output_folder, img_file)
    Image.fromarray(result_img).save(out_path)

    if i % 50 == 0:
        print(f"Processed {i+1}/{len(image_files)} frames")

print("✅ All detections saved to:", output_folder)