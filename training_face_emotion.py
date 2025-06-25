import os
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import ModelCheckpoint

# Paths
train_dir = r"D:\Python Projects\AI Based Face & Emotion Detection Based Attendance System\Emotion Dataset\train"
val_dir = r"D:\Python Projects\AI Based Face & Emotion Detection Based Attendance System\Emotion Dataset\valid"

# Parameters
img_height, img_width = 100, 100
batch_size = 32
num_classes = len(os.listdir(train_dir))
print(num_classes)

# Data Generators
train_datagen = ImageDataGenerator(rescale=1.0 / 255)
val_datagen = ImageDataGenerator(rescale=1.0 / 255)

train_gen = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

val_gen = val_datagen.flow_from_directory(
    val_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical"
)

# Save class indices
import json
with open("emotion_labels.json", "w") as f:
    json.dump(train_gen.class_indices, f)

# Model
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(img_height, img_width, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.3),
    Dense(num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Checkpoint
checkpoint = ModelCheckpoint("face_emotion_model.keras", monitor="val_accuracy", save_best_only=True)

# Train
history = model.fit(
    train_gen,
    epochs=18,
    validation_data=val_gen,
    callbacks=[checkpoint]
)