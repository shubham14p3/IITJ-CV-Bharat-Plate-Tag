# IITJ-BHARAT-NUMBER-PLATE-TAG  
Bharat Number Plate Tag: Multi-Approach Vehicle Plate Detection using YOLO, CNN, and OCR

[![Contributors][contributors-shield]][contributors-url]  
[![Forks][forks-shield]][forks-url]  
[![Stargazers][stars-shield]][stars-url]  
![Issues][issues-shield]

Live Demo: [https://abc.com](https://abc.com)

---

## 🧭 Overview

This project is part of a Computer Vision capstone at **IIT Jodhpur**, aimed at building a robust number plate recognition system for Indian traffic scenarios. It fuses multiple approaches: Deep Learning (YOLOv8, CNN), Optical Character Recognition (Tesseract OCR), and traditional Computer Vision (Canny, Morphology, Color Segmentation). The final result is a Streamlit-based web app capable of real-time and batch image/video analysis.

---

## 🔧 Setup Instructions

### ⚙️ Python Environment Setup

```bash
python -m venv env
# Activate Virtual Environment:
source env/Scripts/activate        # Git Bash
.\env\Scripts\Activate.ps1         # PowerShell
```

### 📦 Install Required Packages

```bash
pip install streamlit opencv-python-headless ultralytics numpy pillow tensorflow matplotlib pytesseract scikit-learn filterpy openpyxl 
pip freeze > requirements.txt
```

---

## 🧠 Model Training & Usage

### YOLOv8: Indian Plate Detector

- **Training File**: `indian_plate.yaml`

```yaml
# Format for YOLOv8 training
dataset:
  train: dataset/images/train
  val: dataset/images/val
nc: 1
names: ["license_plate"]
```

```bash
yolo task=detect mode=train model=yolov8n.pt data=indian_plate.yaml epochs=50
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=indian_plate.yaml
```

- **Output model**: `models/iitj_cv_bharat_plate.pt`

### CNN Binary Classifier: Plate / No Plate

- Files:
  - `generate_no_plate.py`: Creates dummy non-plate images using noise
  - `train_cnn.py`: Trains CNN on labeled plate/no_plate images

```bash
python generate_no_plate.py
python train_cnn.py
```

#### `train_cnn.py` Highlights:

- Uses **ImageDataGenerator** for image augmentation
- CNN with 3 convolution layers, binary sigmoid output
- Saved models:
  - `cnn_plate_classifier_best.h5`
  - `cnn_plate_classifier_latest.h5`
- Graphs: `training_metrics.png`

```python
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=10, zoom_range=0.1,
    width_shift_range=0.1, height_shift_range=0.1,
    shear_range=0.1, horizontal_flip=True)

val_datagen = ImageDataGenerator(rescale=1./255)
```

---

## 🗂 Project Structure

```
BHARAT-NUMBER-PLATE-TAG/
├── data/
│   └── database.db
├── env/
├── images/
│   └── plate_template.png
├── models/
│   ├── iitj_cv_bharat_plate.pt
│   ├── cnn_plate_classifier_best.h5
│   ├── cnn_plate_classifier_latest.h5
│   └── yolov8n.pt
├── assets/
│   └── report.pdf  <-- [Dummy Report](https://abc.com/zbc.pdf)
├── resources/
│   └── B Traffic_O.ttf
├── settings/
│   └── settings.py
├── src/
│   ├── app.py
│   ├── SQLManager.py
│   ├── PlateGen.py
│   ├── plate_reader.py
│   └── sort.py
├── cnn_classifier_data/
│   ├── train/with_plate/, no_plate/
│   └── val/with_plate/, no_plate/
├── videos/
│   └── test.mp4
├── README.md
├── requirements.txt
├── setup.sh
└── webapp.sh
```

---

## 🔍 Detection Modes Explained

1. **YOLOv8 (Deep Learning)**  
   - Pretrained `yolov8n.pt` adapted for Indian plates
   - Output: Bounding boxes with confidence scores

2. **Traditional CV (Canny + Contours)**  
   - Uses edge detection and contour approximation
   - Lightweight and effective in good lighting

3. **Color Segmentation (HSV)**  
   - HSV color space filtering for yellow/white plates
   - Fast, good in controlled environments

4. **Edge + Morph Filter**  
   - Morphological operations after edge detection
   - Enhances plate boundary separation

5. **CNN Classifier (Custom DL)**  
   - Custom Keras CNN for binary classification: plate vs. no_plate
   - Useful for filtering false detections

6. **OCR Plate Recognition**  
   - Tesseract OCR for text extraction
   - Output is stylized using plate template overlays

---

## 🚀 Running the Application

1. Activate the environment:
```bash
source env/Scripts/activate
```

2. Launch the Web App:
```bash
streamlit run app.py
```

3. Optional: Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract/releases)

---

## 📸 Screenshots

| Login UI | Detection Output | OCR Recognition |
|----------|------------------|------------------|
| ![Login](assets/login.JPG) | ![Detected](assets/clean_data.JPG) | ![OCR](assets/data_from_backend.JPG) |

---

## 🧪 Technologies Used

### Backend
- **Flask**
- **Tesseract OCR**
- **YOLOv8 (Ultralytics)**
- **TensorFlow + Keras**
- **OpenCV**
- **SQLite**

### Frontend
- **Streamlit**
- **HTML + CSS Styling**
- **Base64 Image Embeds**
- **Interactive Sliders / Toggles**

---

## 🧑‍💻 Team

| Roll No | Name | Email |
|---------|------|-------|
| **M24DE3076** | Shubham Raj | m24de3076@iitj.ac.in |
| **M24DE3022** | Bhavesh Arora | m24de3022@iitj.ac.in |
| **M24DE3043** | Kanishka Dhindhwal | m24de3043@iitj.ac.in |
| **M24DE3062** | Pratyush Solanki | m24de3062@iitj.ac.in |

---

## 🌱 Future Enhancements

- Enhance OCR accuracy with LSTM-based sequence recognition
- Incorporate vehicle metadata detection (type, color, region)
- Streamline deployment to cloud via Docker or EC2
- Support multi-language plate decoding

---

## 📢 Acknowledgements

- Supported by [IIT Jodhpur](https://www.iitj.ac.in/)
- Dataset inspired by Indian Traffic Scene data
- YOLOv8 provided by [Ultralytics](https://github.com/ultralytics)

---

## 🌟 Show Your Support

Give a ⭐ on [GitHub](https://github.com/shubham14p3) if you found this useful.

---

<!-- MARKDOWN LINKS & BADGES -->
[contributors-shield]: https://img.shields.io/github/contributors/shubham14p3/IITJ-BHARAT-NUMBER-PLATE-TAG.svg?style=flat-square  
[contributors-url]: https://github.com/shubham14p3/IITJ-BHARAT-NUMBER-PLATE-TAG/graphs/contributors  
[forks-shield]: https://img.shields.io/github/forks/shubham14p3/IITJ-BHARAT-NUMBER-PLATE-TAG.svg?style=flat-square  
[forks-url]: https://github.com/shubham14p3/IITJ-BHARAT-NUMBER-PLATE-TAG/network/members  
[stars-shield]: https://img.shields.io/github/stars/shubham14p3/IITJ-BHARAT-NUMBER-PLATE-TAG.svg?style=flat-square  
[stars-url]: https://github.com/shubham14p3/IITJ-BHARAT-NUMBER-PLATE-TAG/stargazers  
[issues-shield]: https://img.shields.io/github/issues/shubham14p3/IITJ-BHARAT-NUMBER-PLATE-TAG.svg?style=flat-square
