# src/cnn_plate_pipeline.py

import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score
import tensorflow.keras.backend as K
import streamlit as st

# -------------------------------
# Function: iitj_cv_bharat_plate
# -------------------------------
def detect_license_plate(image, plate_text=''):
    cascade_path = 'models/iitj_cv_bharat_plate.xml'
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    image_copy = image.copy()
    detected_plate = None

    plate_rects = plate_cascade.detectMultiScale(image_copy, scaleFactor=1.2, minNeighbors=7)
    for (x, y, w, h) in plate_rects:
        roi = image_copy[y:y+h, x:x+w]
        detected_plate = roi.copy()
        cv2.rectangle(image_copy, (x+2, y), (x+w-3, y+h-5), (51, 181, 155), 3)
        if plate_text:
            cv2.putText(image_copy, plate_text, (x - w//2, y - h//2),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.5, (51, 181, 155), 1, cv2.LINE_AA)
    return image_copy, detected_plate

# -----------------------------------
# Function: find_contours (for debugging)
# -----------------------------------
def find_contours(dimensions, img):
    cntrs, _ = cv2.findContours(img.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    lower_width, upper_width, lower_height, upper_height = dimensions
    # Keep only the largest 15 contours
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)[:15]
    x_cntr_list = []
    char_images = []
    
    # Debug image for visualizing contours
    debug_img = cv2.cvtColor(img.copy(), cv2.COLOR_GRAY2BGR)

    for cnt in cntrs:
        x, y, w, h = cv2.boundingRect(cnt)
        if lower_width < w < upper_width and lower_height < h < upper_height:
            x_cntr_list.append(x)
            char = img[y:y+h, x:x+w]
            char = cv2.resize(char, (20, 40))
            char = cv2.subtract(255, char)
            char_canvas = np.zeros((44, 24), dtype=np.uint8)
            char_canvas[2:42, 2:22] = char
            char_images.append(char_canvas)
            cv2.rectangle(debug_img, (x, y), (x+w, y+h), (0,255,0), 2)
    
    # Display debugging image using matplotlib (embedded in Streamlit)
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.imshow(cv2.cvtColor(debug_img, cv2.COLOR_BGR2RGB))
    ax.set_title("Contours Debug")
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

    # Sort the character images based on x-coordinate (left-to-right order)
    sorted_indices = sorted(range(len(x_cntr_list)), key=lambda i: x_cntr_list[i])
    sorted_chars = [char_images[i] for i in sorted_indices]

    return np.array(sorted_chars)

# -----------------------------------
# Function: segment_characters
# -----------------------------------
def segment_characters(plate_img):
    # Resize the cropped plate image to a fixed size
    img_lp = cv2.resize(plate_img, (333, 75))
    img_gray_lp = cv2.cvtColor(img_lp, cv2.COLOR_BGR2GRAY)
    _, img_binary_lp = cv2.threshold(img_gray_lp, 200, 255,
                                     cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img_binary_lp = cv2.erode(img_binary_lp, (3, 3))
    img_binary_lp = cv2.dilate(img_binary_lp, (3, 3))

    # Force white borders
    img_binary_lp[0:3, :] = 255
    img_binary_lp[:, 0:3] = 255
    img_binary_lp[72:75, :] = 255
    img_binary_lp[:, 330:333] = 255

    # Debug: Show the binary plate image using matplotlib
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.imshow(img_binary_lp, cmap='gray')
    ax.set_title("Binary Plate Image")
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig)

    # Define contour size limits: [min_width, max_width, min_height, max_height]
    dimensions = [img_binary_lp.shape[0] / 6,
                  img_binary_lp.shape[0] / 2,
                  img_binary_lp.shape[1] / 10,
                  2 * img_binary_lp.shape[1] / 3]

    char_list = find_contours(dimensions, img_binary_lp)

    # Debug: Show each segmented character in a subplot
    if len(char_list) > 0:
        fig, axs = plt.subplots(1, len(char_list), figsize=(len(char_list) * 2, 2))
        if len(char_list) == 1:
            axs.imshow(char_list[0], cmap='gray')
            axs.set_title("Char 1")
            axs.axis('off')
        else:
            for i, ch in enumerate(char_list):
                axs[i].imshow(ch, cmap='gray')
                axs[i].set_title(f"Char {i+1}")
                axs[i].axis('off')
        st.pyplot(fig)
        plt.close(fig)
    return char_list

# -----------------------------------
# Function: fix_dimension (for character images)
# -----------------------------------
def fix_dimension(img, target_size):
    """
    Resizes a grayscale character image to the target size and replicates it to 3 channels.
    target_size: tuple (height, width) extracted from model.input_shape.
    Returns a float32 image normalized to [0,1].
    """
    # Resize image to target dimensions (cv2.resize expects (width, height))
    img_resized = cv2.resize(img, (target_size[1], target_size[0]), interpolation=cv2.INTER_AREA)
    # If the image is grayscale (2D), convert it to 3-channel image
    if len(img_resized.shape) == 2:
        img_resized = np.stack([img_resized] * 3, axis=-1)
    elif img_resized.shape[2] == 1:
        img_resized = np.concatenate([img_resized]*3, axis=-1)
    return img_resized.astype(np.float32) / 255.0

def fix_dimension(img):
    """
    Resizes a grayscale character image to 28×28 and replicates it to 3 channels.
    This ensures that no matter what size is returned by segmentation,
    the final image sent to the model is exactly 28x28x3.
    """
    # Resize to 28x28 using INTER_AREA for good quality when downsizing/upscaling
    resized = cv2.resize(img, (28, 28), interpolation=cv2.INTER_AREA)
    # If the image is grayscale (2D), duplicate the channel to make it 3-channel.
    if len(resized.shape) == 2:
        resized = np.stack([resized] * 3, axis=-1)
    elif resized.shape[2] == 1:
        resized = np.concatenate([resized] * 3, axis=-1)
    return resized.astype(np.float32) / 255.0

def predict_plate_number(model, char_imgs):
    """
    For each segmented character image, resize it to 28x28 with fix_dimension,
    then run it through the CNN model to predict the character. It then concatenates
    all predictions into the final plate number.
    """
    # Mapping of class indices to characters
    char_dict = {i: ch for i, ch in enumerate('0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ')}
    predicted_chars = []
    for ch_img in char_imgs:
        # Resize segmented character to 28x28 and convert to 3-channel normalized float32 image
        processed = fix_dimension(ch_img)
        # Add batch dimension: shape (1,28,28,3)
        processed = processed.reshape(1, 28, 28, 3)
        preds = model.predict(processed)
        pred_idx = int(np.argmax(preds))
        predicted_chars.append(char_dict[pred_idx])
    return ''.join(predicted_chars)


# -----------------------------------
# Custom F1 Metric (for model loading)
# -----------------------------------
@tf.keras.utils.register_keras_serializable()
def f1_metric(y_true, y_pred):
    return f1_score(y_true, tf.argmax(y_pred, axis=1), average='micro')

def custom_f1(y_true, y_pred):
    return tf.py_function(f1_metric, (y_true, y_pred), tf.double)

# -----------------------------------
# Function: load_or_train_cnn_model
# -----------------------------------
@st.cache_resource
def load_or_train_cnn_model(model_path='models/plate_classifier.h5'):
    """
    Loads the character recognition model.
    Note: This model was trained on images resized to (28,28,3). 
    Make sure that you have trained the model using your training dataset,
    which is located under models/Training/data.
    """
    model = load_model(model_path, custom_objects={'custom_f1': custom_f1})
    return model

# -----------------------------------
# Exported Functions
# -----------------------------------
__all__ = [
    "detect_license_plate",
    "segment_characters",
    "predict_plate_number",
    "load_or_train_cnn_model"
]
