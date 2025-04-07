# IITJ-BHARAT-NUMBER-PLATE-TAG  
Bharat Number Plate Tag: Multi-Approach Vehicle Plate Detection using YOLO, CNN, and OCR

[![Contributors][contributors-shield]][contributors-url]  
[![Forks][forks-shield]][forks-url]  
[![Stargazers][stars-shield]][stars-url]  
![Issues][issues-shield]

---

## 🧭 Overview

This project is part of a Computer Vision capstone at **IIT Jodhpur**, focused on robust number plate detection and tagging in Indian traffic scenarios. It combines modern Deep Learning (YOLOv8, CNN), Optical Character Recognition (Tesseract OCR), and traditional Computer Vision techniques. The end result is a fully interactive **Streamlit-based Web Application** with image/video support, plate recognition, and plate tagging.

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
pip install streamlit opencv-python-headless ultralytics numpy pillow tensorflow matplotlib pytesseract scikit-learn filterpy
pip freeze > requirements.txt
```

### 🧠 Model Training & Usage

#### YOLOv8 Indian Plate Detector

```bash
yolo task=detect mode=train model=yolov8n.pt data=indian_plate.yaml epochs=50
yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=indian_plate.yaml
```

Move the best model to models directory:
```
models/indian_plate_detection.pt
```

#### CNN Classifier

```bash
python generate_no_plate.py
python train_cnn.py
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
│   ├── indian_plate_detection.pt
│   └── cnn_plate_classifier.h5
├── resources/
│   └── B Traffic_O.ttf
├── settings/
│   └── settings.py
├── src/
│   ├── app.py
│   ├── sort.py
│   ├── SQLManager.py
│   ├── PlateGen.py
│   └── plate_reader.py
├── videos/
│   └── test.mp4
├── cnn_classifier_data/
│   ├── train/
│   │   ├── with_plate/
│   │   └── no_plate/
│   └── val/
│       ├── with_plate/
│       └── no_plate/
├── README.md
├── requirements.txt
├── setup.sh
└── webapp.sh
```

---

## 🔍 Detection Modes Supported

- YOLOv8 Deep Learning Detector  
- Traditional Canny + Contours CV  
- HSV Color Segmentation  
- Morphology-based Edge Filter  
- Custom CNN Binary Classifier  
- OCR-based Plate Recognition with Stylized Output

---

## 🚀 Running the Application

1. Activate the environment:
```bash
source env/Scripts/activate
```

2. Run the Streamlit Web App:
```bash
streamlit run src/app.py
```

3. Optional: Install [Tesseract OCR](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)

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