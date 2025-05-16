import os
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# === CONFIG ===
tracked_frames_dir = "tracked_frames"
field_image_path = "field.jpg"  # Must match the tracked frame resolution
output_dir = "field_heatmaps0"
os.makedirs(output_dir, exist_ok=True)

# === Manually Map Tracked IDs to Player Labels ===
player_id_map = {
    "TeamA_1": 12,
    "TeamA_2": 9,
    "TeamA_3": 10,
    "TeamB_1": 8,
    "TeamB_2": 13,
    "TeamB_3": 18
}


# === Load Field Background ===
field_img = cv2.imread(field_image_path)
if field_img is None:
    raise FileNotFoundError("❌ Cannot load field image. Please check 'field.jpg' exists.")
field_img = cv2.cvtColor(field_img, cv2.COLOR_BGR2RGB)
field_h, field_w = field_img.shape[:2]

# === Parse Bounding Boxes and Track IDs from Frames ===
player_positions = {label: [] for label in player_id_map}
frame_files = sorted(f for f in os.listdir(tracked_frames_dir) if f.endswith(".jpg"))

for fname in frame_files:
    frame_path = os.path.join(tracked_frames_dir, fname)
    frame = cv2.imread(frame_path)

    if frame is None:
        continue

    for label, id_num in player_id_map.items():
        text = f"ID {id_num}"
        found = False

        for y in range(0, frame.shape[0] - 10, 5):
            for x in range(0, frame.shape[1] - 50, 5):
                roi = frame[y:y + 12, x:x + 50]
                if roi.shape[0] < 12 or roi.shape[1] < 50:
                    continue

                roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                _, roi_thresh = cv2.threshold(roi_gray, 200, 255, cv2.THRESH_BINARY)
                roi_text = cv2.putText(np.zeros_like(roi_gray), text, (0, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255, 1)
                match = np.mean((roi_thresh == roi_text).astype(np.uint8))

                if match > 0.9:
                    center_x, center_y = x + 25, y + 6
                    player_positions[label].append((center_x, center_y))
                    found = True
                    break
            if found:
                break

# === Generate KDE Heatmaps ===
for label, coords in player_positions.items():
    if len(coords) < 5:
        print(f"⚠️ Not enough data for {label}")
        continue

    x, y = zip(*coords)
    plt.figure(figsize=(12, 7))

    # Add 10% visual margin
    margin_x = int(field_w * 0.05)
    margin_y = int(field_h * 0.05)

    plt.imshow(field_img, extent=[-margin_x, field_w + margin_x, field_h + margin_y, -margin_y])
    sns.kdeplot(
        x=x,
        y=y,
        shade=True,
        cmap="Reds",
        alpha=0.6,
        bw_adjust=0.5,
        clip=((0, field_w), (0, field_h)),  # Keep heat within field
        thresh=0.05,
    )

    plt.title(f"Heatmap: {label}")
    plt.xlim(-margin_x, field_w + margin_x)
    plt.ylim(field_h + margin_y, -margin_y)
    plt.axis("off")

    out_path = os.path.join(output_dir, f"{label}_heatmap.png")
    plt.savefig(out_path, bbox_inches="tight", pad_inches=0)
    plt.close()

    print(f"✅ Saved: {out_path}")