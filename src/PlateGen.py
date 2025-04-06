# src/PlateGen.py
import cv2
import numpy as np

# Function to generate a stylized image of the plate text
def PlateGen(plate_text):
    width, height = 320, 80
    plate = np.ones((height, width, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.8
    thickness = 3
    text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    cv2.putText(plate, plate_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    return plate
