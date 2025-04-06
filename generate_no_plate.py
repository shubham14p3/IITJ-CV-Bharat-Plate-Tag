import os
import numpy as np
import cv2

# Folder to save no_plate images
train_dir = "cnn_classifier_data/train/no_plate"
val_dir = "cnn_classifier_data/val/no_plate"

# Create folders
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)

# Generate 10 random dummy images for training
for i in range(10):
    img = np.random.randint(0, 100, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(train_dir, f"random_train_{i}.jpg"), img)

# Generate 5 for validation
for i in range(5):
    img = np.random.randint(0, 100, (64, 64, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(val_dir, f"random_val_{i}.jpg"), img)

print("✅ Dummy no_plate images created successfully.")
