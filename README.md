# IITJ-CV-Bharat-Plate-Tag
python -m venv env

source env/Scripts/activate (For Git Bash Only)
.\env\Scripts\Activate.ps1 (For Powershell)

pip install streamlit opencv-python-headless ultralytics numpy pillow tensorflow matplotlib scikit-learn filterpy
pip freeze > requirements.txt

yolo task=detect mode=train model=yolov8n.pt data=indian_plate.yaml epochs=50


yolo task=detect mode=val model=runs/detect/train/weights/best.pt data=indian_plate.yaml
Export:
Copy the best model file (usually best.pt from runs/detect/train/weights/) into your project’s models folder and rename it (for example, to indian_plate_detection.pt).
python generate_no_plate.py
python train_cnn.py
PROJECT
 ├── data
 │     └── database.db
 ├── env
 ├── images
 │     └── plate_template.png   # if needed for UI or plate generation (not used for detection)
 ├── models
 │     ├── indian_plate_detection.pt  # your trained YOLO model
 ├── resources
 │     └── B Traffic_O.ttf
 ├── settings
 │     └── settings.py
 ├── src
 │     ├── app.py                   # Main Streamlit app
 │     ├── sort.py                  # Tracker code (if you need tracking)
 │     ├── SQLManager.py            # For database handling
 │     ├── PlateGen.py              # (Optional) For generating plate images
 │     └── plate_reader.py          # (Not used since OCR is not allowed)
 ├── videos
 │     └── test.mp4
 ├── README.md
 ├── requirements.txt
 ├── setup.sh
 └── webapp.sh

cnn_classifier_data/
├── train/
│   ├── with_plate/      <- Your real samples
│   └── no_plate/        <- Dummy images (auto-generated)
├── val/
│   ├── with_plate/
│   └── no_plate/
models/
└── cnn_plate_classifier.h5
source env/Scripts/activate
streamlit run app.py


