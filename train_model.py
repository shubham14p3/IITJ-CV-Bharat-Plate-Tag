# train_model.py

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras import optimizers
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K
from sklearn.metrics import f1_score

# Define custom f1 metric for monitoring training (if desired)
def f1score(y, y_pred):
    return f1_score(y, tf.math.argmax(y_pred, axis=1), average='micro') 

def custom_f1score(y, y_pred):
    return tf.py_function(f1score, (y, y_pred), tf.double)

K.clear_session()

# Data generators: adjust paths according to your folder structure
data_path = os.path.join("models", "Training", "data")
train_dir = os.path.join(data_path, "train")
val_dir = os.path.join(data_path, "val")

train_datagen = ImageDataGenerator(rescale=1./255, width_shift_range=0.1, height_shift_range=0.1)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(28,28), 
    batch_size=1,
    class_mode='sparse'
)

validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(28,28),
    batch_size=1,
    class_mode='sparse'
)

# Build the CNN model using your architecture
model = Sequential()
model.add(Conv2D(16, (22,22), input_shape=(28,28,3), activation='relu', padding='same'))
model.add(Conv2D(32, (16,16), activation='relu', padding='same'))
model.add(Conv2D(64, (8,8), activation='relu', padding='same'))
model.add(Conv2D(64, (4,4), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(4,4)))
model.add(Dropout(0.4))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(36, activation='softmax'))

# Compile the model; you can also add custom_f1score in metrics if desired
model.compile(
    loss='sparse_categorical_crossentropy', 
    optimizer=optimizers.Adam(learning_rate=0.0001), 
    metrics=['accuracy']  # or metrics=[custom_f1score] for your custom metric
)

model.summary()

# Optionally, add early stopping to prevent overfitting
early_stop = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)

# Train the model; steps_per_epoch is defined by total samples / batch_size
model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 1,
    validation_data=validation_generator,
    epochs=80,
    callbacks=[early_stop],
    verbose=1
)

# Save the trained model to models/cnn_plate_classifier1.h5
model_save_path = os.path.join("models", "c.h5")
model.save(model_save_path)
print("Model training complete. Saved at:", model_save_path)
