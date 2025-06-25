import cv2
import numpy as np
from tensorflow.keras.models import load_model
import json
import csv
import os
from datetime import datetime

csv_filename = r"D:\Python Projects\AI Based Face & Emotion Detection Based Attendance System\attendance_log.csv"

# Create CSV file with headers if not exists
if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date_Time", "Emotion"])

# Load model and labels
face_recognition_model = load_model("face_recognition_model.keras")
face_emotion_model = load_model("face_emotion_model.keras")

with open("class_labels.json", "r") as f:
    label_map = json.load(f)
rev_map = {v: k for k, v in label_map.items()}  # {0: "Akshay", 1: "Virat", ...}

with open("emotion_labels.json", "r") as f:
    label_map = json.load(f)
emotion_rev_map = {v: k for k, v in label_map.items()}  # {0: "Akshay", 1: "Virat", ...}

# Face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Define face size threshold (width in pixels)
MIN_FACE_WIDTH = 100  # Adjust this threshold for sensitivity

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale and detect face
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
        # Check face size (indicates closeness to camera)
        if w >= MIN_FACE_WIDTH:
            # Face is near enough — proceed
            face = rgb_frame[y:y+h, x:x+w]
            face = cv2.resize(face, (100, 100))
            # face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = face / 255.0
            face = np.expand_dims(face, axis=0)

            pred = face_recognition_model.predict(face)[0]
            class_id = np.argmax(pred)
            confidence = pred[class_id]
            # Set a confidence threshold
            name = rev_map[class_id] if confidence > 0.9 else "Unknown"
            
            # EMOTION DETECTION
            emotion = rgb_frame[y:y+h, x:x+w]
            emotion_resized = cv2.resize(emotion, (100, 100))
            emotion_normalized = emotion_resized / 255.0
            emotion_input = np.expand_dims(emotion_normalized, axis=0)
            # face_gray_input = np.expand_dims(face_gray_normalized, axis=-1)  # (1,100,100,1)

            pred_emotion = face_emotion_model.predict(emotion_input)[0]
            emotion_id = np.argmax(pred_emotion)
            emotion_conf = pred_emotion[emotion_id]
            emotion_label = emotion_rev_map[emotion_id] if emotion_conf > 0.7 else "No emotion"

            # Draw results
            label = f"{name} ({confidence*100:.1f}%) | {emotion_label} ({emotion_conf*100:.0f}%)"
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            cv2.putText(frame, label, (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            
            if name != "Unknown":
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                with open(csv_filename, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow([name, current_time, emotion_label])
        else:
            # Face too far — show guidance
            cv2.putText(frame, "Come Closer", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)

    cv2.imshow("Real-Time Face Recognition & Emotion Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
