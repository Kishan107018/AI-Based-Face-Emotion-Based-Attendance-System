import os, shutil
import random

source_dir = "D:\Python Projects\AI Based Face & Emotion Detection Based Attendance System\Face Identification Dataset"
train_dir = "train"
val_dir = "valid"

split_ratio = 0.7  # 80% train, 20% val

for person in os.listdir(source_dir):
    files = os.listdir(os.path.join(source_dir, person))
    random.shuffle(files)
    split = int(len(files) * split_ratio)

    train_files = files[:split]
    val_files = files[split:]

    os.makedirs(os.path.join(train_dir, person), exist_ok=True)
    os.makedirs(os.path.join(val_dir, person), exist_ok=True)

    for file in train_files:
        shutil.copy(os.path.join(source_dir, person, file),
                    os.path.join(train_dir, person, file))
    for file in val_files:
        shutil.copy(os.path.join(source_dir, person, file),
                    os.path.join(val_dir, person, file))
