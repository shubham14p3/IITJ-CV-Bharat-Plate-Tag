import cv2
import numpy as np
import os

def PlateGen(plate_text, left_margin=150):
    """
    Generates a license plate image given the plate_text by overlaying text on a template.
    If any error occurs, a fallback blank plate image with an error message is returned.
    
    Parameters:
        plate_text (str): The text to display on the plate.
        left_margin (int): Additional left margin (in pixels) for the generated image.
    
    Returns:
        np.ndarray: The generated license plate image.
    """
    try:
        # Ensure plate_text is a string.
        plate_text = str(plate_text) if plate_text is not None else ''
        
        # Define the path to your template image.
        template_path = "assets/plate_template.png"
        if not os.path.exists(template_path):
            raise FileNotFoundError(f"Plate template not found at {template_path}")
        
        # Load the template image.
        plate = cv2.imread(template_path)
        if plate is None:
            raise ValueError("Failed to load plate template image with cv2.imread.")
        
        # Get the dimensions of the template image.
        height, width, channels = plate.shape

        # Create a new image with an added left margin.
        new_width = width + left_margin
        new_plate = np.ones((height, new_width, channels), dtype=np.uint8) * 255  # White background.
        new_plate[:, left_margin:new_width] = plate

        # Set font properties.
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 5
        thickness = 15
        
        # Determine text size for centering the text within the original plate area.
        text_size = cv2.getTextSize(plate_text, font, font_scale, thickness)[0]
        text_x = left_margin + (width - text_size[0]) // 2
        text_y = (height + text_size[1]) // 2
        
        # Write the plate text onto the new image.
        cv2.putText(new_plate, plate_text, (text_x, text_y), font, font_scale, (0, 0, 0), thickness)

        return new_plate

    except Exception as e:
        # Log the error message (you can change print to your own logging method).
        print("Error in PlateGen:", e)
        
        # Create a fallback image: a simple white plate with an error message.
        fallback_height = 150
        fallback_width = 500
        fallback_plate = np.ones((fallback_height, fallback_width, 3), dtype=np.uint8) * 255  # White background.
        fallback_message = "Error: " + str(e)
        
        # Use a basic font for the error message.
        fallback_font = cv2.FONT_HERSHEY_SIMPLEX
        fallback_font_scale = 0.7
        fallback_thickness = 2
        
        try:
            error_text_size = cv2.getTextSize(fallback_message, fallback_font, fallback_font_scale, fallback_thickness)[0]
            text_x = (fallback_width - error_text_size[0]) // 2
            text_y = (fallback_height + error_text_size[1]) // 2
            cv2.putText(fallback_plate, fallback_message, (text_x, text_y), fallback_font, fallback_font_scale, (0, 0, 255), fallback_thickness)
        except Exception as inner_e:
            # If an error occurs while generating the error message image, simply pass.
            print("Error while generating fallback plate text:", inner_e)
        
        return fallback_plate
