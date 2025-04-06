import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Paths
train_path = "cnn_classifier_data/train"
val_path = "cnn_classifier_data/val"

# Data Loaders
train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_set = train_datagen.flow_from_directory(train_path, target_size=(64, 64), batch_size=32, class_mode='binary')
val_set = val_datagen.flow_from_directory(val_path, target_size=(64, 64), batch_size=32, class_mode='binary')

# Model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)),
    layers.MaxPooling2D(2, 2),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(2, 2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Training
model.fit(train_set, epochs=10, validation_data=val_set)

# Save model
model.save("models/cnn_plate_classifier.h5")
