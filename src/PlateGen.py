import cv2
import numpy as np
import os

def PlateGen(plate_text, left_margin=150):
    # Path to your template image (adjust the path as needed)
    template_path = "assets/plate_template.png"
    
    # Check if the file exists
    if not os.path.exists(template_path):
        raise FileNotFoundError(f"Plate template not found at {template_path}")
    
    # Load the template image
    plate = cv2.imread(template_path)
    
    # Get the dimensions of the template image
    height, width, channels = plate.shape
    
    # Create a new image with added left margin.
    # Here we add a white (255,255,255) margin of left_margin pixels.
    new_width = width + left_margin
    # Create a blank (white) image of the new size.
    new_plate = np.ones((height, new_width, channels), dtype=np.uint8) * 255
    # Copy the original plate image into the new image, starting at column 'left_margin'
    new_plate[:, left_margin:new_width] = plate
    
    # Set font properties
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 5
    thickness = 15
    
    # Determine text size for centering the text within the original plate area
    text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
    # The text is centered in the plate region only (not across the white margin)
    # Calculate x such that the text is centered from left_margin to new_width
    text_x = left_margin + (width - text_size[0]) // 2
    text_y = (height + text_size[1]) // 2
    
    # Write the plate text onto the new image
    cv2.putText(new_plate, plate_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)
    
    return new_plate
