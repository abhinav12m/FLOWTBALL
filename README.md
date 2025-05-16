# ⚽️ Soccer Player Tracking with YOLOv8 + DeepSORT

This project implements an end-to-end pipeline to extract soccer player movement patterns from match footage. It uses **YOLOv8** for player detection, **DeepSORT** for ID tracking, and **Seaborn** to generate player-specific heatmaps over a soccer field.

---

## 🚀 Features

* Frame extraction from video at 15 FPS
* Player detection using pretrained YOLOv8
* Multi-object tracking with DeepSORT (frames 130 to 400)
* Manual ID labeling and heatmap generation using KDE

---

## 📆 Step-by-Step Setup & Execution

### ✅ 1. Clone Repositories

```bash
git clone https://github.com/ultralytics/ultralytics yolo-ultralytics
cd yolo-ultralytics
pip install .  # Installs YOLOv8
cd ..

git clone https://github.com/ZQPei/deep_sort_pytorch deepsort
```

---

### ✅ 2. Create Python 3.10 Virtual Environment

```bash
python3.10 -m venv venv
source venv/bin/activate   # or venv\Scripts\activate on Windows
```

### ✅ 3. Install Required Packages

```bash
pip install -r requirements.txt
```

**`requirements.txt`**

```
ultralytics==8.0.197
torch>=1.12
opencv-python
matplotlib
seaborn
numpy
pandas
scikit-learn
```

---

### 📁 Note on Input Video

> **⚠️ GitHub File Size Limit**

The original match video `Input.mp4` exceeds GitHub’s 100 MB upload limit and cannot be stored in this repository.

📥 **Download it from Google Drive:**
[🔗 Click here to access Input.mp4](https://drive.google.com/file/d/1iBtlb8SkjRaJe8l0Nu7ZEvdlawlHUeWh/view?usp=drive_link)
*(Replace this link with your actual Google Drive link)*

Once downloaded, place the file in the root directory of the project (same level as `extract_frames.py`) to ensure compatibility with the pipeline.

```bash
project/
├── Input.mp4  👈 Place it here
├── extract_frames.py
├── ...
```

---


## 🔹 Pipeline Execution

### ✅ Step 1: Extract Frames (15 FPS)

```bash
python extract_frames.py
```

* Input: `Input.mp4`
* Output: `frames/frame_0000.jpg`, ...

### ✅ Step 2: Run YOLOv8 Detection (All Frames)

```bash
python detect_yolo.py
```

* Input: `frames/`
* Output: `detections/`

### ✅ Step 3: Track Players using DeepSORT (Frames 130–400)

```bash
python run_tracker.py
```

* Input: `frames/`
* Output: `tracked_frames/` with bounding boxes + IDs

> ⚡ To change tracking range, update:

```python
if idx < 130 or idx > 400:
    continue
```

To:

```python
if idx < START or idx > END:
    continue
```

### ✅ Step 4: Generate Heatmaps

```bash
python generate_heatmap.py
```

* Input: `tracked_frames/`, `field.jpg`
* Output: `field_heatmaps0/TeamA_1_heatmap.png`, ...

> Edit `player_id_map` in `generate_heatmap.py` to map **player names** to **DeepSORT track IDs**:

```python
player_id_map = {
    "TeamA_1": 12,
    "TeamA_2": 9,
    "TeamA_3": 10,
    "TeamB_1": 8,
    "TeamB_2": 13,
    "TeamB_3": 18
}
```

> ⚠ If you rerun tracking and IDs change, update the IDs in the map accordingly.

---

## 📁 Project Structure

```
.
├── Input.mp4
├── frames/                   # Extracted frames
├── detections/               # YOLOv8 bounding box images
├── tracked_frames/           # DeepSORT ID labeled frames
├── field.jpg                 # Soccer field image
├── field_heatmaps0/          # KDE heatmaps
├── extract_frames.py
├── detect_yolo.py
├── run_tracker.py
├── generate_heatmap.py
├── requirements.txt
└── README.md
```

---

## 🔖 References

* Ultralytics. (2023). [YOLOv8](https://github.com/ultralytics/ultralytics)
* ZQPei. (2020). [DeepSORT](https://github.com/ZQPei/deep_sort_pytorch)
* Wojke et al. (2017). *Simple Online and Realtime Tracking with a Deep Association Metric*. [arXiv:1703.07402](https://arxiv.org/abs/1703.07402)
* Zhang et al. (2022). *ByteTrack: Multi-Object Tracking by Associating Every Detection Box*. [arXiv:2110.06864](https://arxiv.org/abs/2110.06864)
* Teed & Deng (2020). *RAFT: Recurrent All Pairs Field Transforms for Optical Flow*. [GitHub](https://github.com/princeton-vl/RAFT)

---# FLOWTBALL
# FLOWTBALL
