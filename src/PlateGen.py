import cv2
import numpy as np
import os

def PlateGen(plate_text):
    # Path to your template image (adjust the path as needed)
    template_path = "assets/plate_template.png"
    
    # Check if the file exists
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Plate template not found at {template_path}")
    
    # Load the template image
    plate = cv2.imread(template_path)
    
    # Get the dimensions of the template
    height, width, _ = plate.shape
    
    # Use a thicker/bolder font than Simplex
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Increase font scale and thickness for a larger, bolder look
    font_scale = 7
    thickness = 15
    
    # Determine the size of the text to center it on the template
    text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
    text_x = (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # Write the plate text onto the template
    cv2.putText(plate, plate_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    return plate
