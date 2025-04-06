# IITJ-CV-Bharat-Plate-Tag
python -m venv env

source env/Scripts/activate
pip install streamlit opencv-python-headless ultralytics numpy pillow tensorflow matplotlib scikit-learn
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

streamlit run app.py




the folder structure
PROJECT
data
- database.db
env
images
- plate_template.png
models
- OCRplate.pt
- platedetection.pt
resources
- B Traffic_O.ttf
settings
> —pycache_
- settings.py
src
- —pycache_
- app.py
- OCRplate.pt
- plate_reader.py
- platedetection.pt
- PlateGen.py
- requirements.txt
- settings.py
- setup.sh
- sort.py
- SQLManager.py
- webapp.sh
videos
- test.mp4
gitignore
app.py
LICENSE
output_deep.jpg
output_traditional.jpg
README.md
requirements.txt
setup.sh
webapp.sh


now i have the following task 
You can choose any topic, but the topic has to be related to Computer Vision. - Bharat Numberplate (IN short it is detecton of Vehicles Number Plate  or Registration Number)
Deep Learning methods are allowed but the topic has to be related to computer vision.
New Findings/Analysis will be appreciated.
OCR based projects are not allowed.
Topics involving Audio, Text, and other non Image data are not allowed. Primary task has to be based on image/video data.
Every group has to apply atleast two different approaches to the selected topic, e.g., a deep learning based solution and a traditional CV based solution.
Ideal case: 4-5 approaches - 1 for each member
Project has to be deployed on online platforms such as streamlit.
The Project should not be the same as your other course projects like for Deep Learning, Digital Image Processing.


THe things i can try to implement can be in these 
Objectives
The Instructor will:
1. Provide insights into fundamental concepts and algorithms behind some of
the remarkable success of Computer Vision
2. Impart working expertise by means of programming assignments and a project
Learning Outcomes
The students are expected to have the ability to:
1. Learn and appreciate the usage and implications of various Computer Vision techniques
in real-world scenarios
2. Design and implement basic applications Of Computer Vision
Course Content
Introduction: The Three R's - Recognition, Reconstruction, Reorganization (1 Lecture)
Fundamentals: Formation, Filtering, Transformation, Alignrnent, Color (5 Lectures)
Image Restoration: Spatial Processing and Wavelet-based Processing (5 Lectures)
Geometry: Homography, Warping, Epipolar Geometry, Stereo, Structure from Motion, Optical flow (9
Lectures)
Segmentation: Key point Extraction, Region Segmentation (e.g., boosting, graph-cut and level-set),
RANSAC (6 Lectures)
Feature Description and Matching: Key-point Description, handcrafted feature extraction (SIFT, LBP)
(3 Lectures)
Deep Learning based Segmentation and Recognition: DL-based Object detection (e.g. Mask-RCNN,
YOLO), Semantic Segmentation, Convolutional Neural Network (CNN) based approaches to visual
recognition (9 Lectures)
Applications: M ultimodal and Multitask Applications (4 Lectures)


pip install ultralytics