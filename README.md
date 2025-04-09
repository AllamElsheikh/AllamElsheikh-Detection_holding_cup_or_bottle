# 🧠 YOLOv5 Object Holding Detection with Tracking

This project detects **if a person is holding a cup or bottle** in images or videos using **YOLOv5** for object detection and **Norfair** for object tracking.  
It's built for AI-based surveillance, behavioral analysis, or smart monitoring systems.

---

## 📂 Project Structure

```bash
.
├── detection_with_tracking.py   # ✅ Main script: Holding detection + tracking
├── models/                      # YOLOv5 model files
├── utils/                       # Helper functions (from YOLOv5)
├── requirements.txt             # Python dependencies
├── outputs/                     # Output images or videos
└── README.md                    # You're here!
 ``` 
## 🔍 Key Features
- 🎯 Detects people, cups, and bottles using YOLOv5

- 🧩 Determines if a person is holding the object

- 📌 Tracks persons over frames using Norfair

- 💾 Saves output video or images with annotations


## 🚀 Getting Started
### 1. Clone the Repo
```bash
git https://github.com/AllamElsheikh/Detection_holding_cup_or_bottle.git
cd Detection_holding_cup_or_bottle/yolov5
```

### 2. Install Dependencies
```bash
!pip install -r requirements.txt

```

## 💡 Detection Logic Highlight
```bash
✅ detection_with_tracking.py
```
## 🛠 TODO

 - Add webcam support

- Extend to detect "holding phone" or "holding bag"

- Integrate alert system

## Auther 
## ALLAM abdelmawgoud . 
