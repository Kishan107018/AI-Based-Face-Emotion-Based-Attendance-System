import cv2, os
import numpy as np
import json
from tensorflow.keras.models import load_model

model = load_model("face_recognition_model.keras")

with open("class_labels.json", "r") as f:
    class_indices = json.load(f)
    
index_to_label = {v: k for k, v in class_indices.items()}

dataset = r'D:\Python Projects\AI Based Face & Emotion Detection Based Attendance System\Face Identification Dataset\valid\Virat Kohli'

for img_path in os.listdir(dataset):
    print(os.path.join(dataset,img_path))
    img = cv2.imread(os.path.join(dataset,img_path))

    # Resize, convert, normalize
    face = cv2.resize(img, (100, 100))
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    face = face / 255.0
    face = np.expand_dims(face, axis=0)

    # Predict
    pred = model.predict(face)
    class_id = np.argmax(pred)
    confidence = pred[0][class_id]
    pred_name = index_to_label[class_id]

    print(f"Prediction: {pred_name} (Confidence: {confidence:.2f})")