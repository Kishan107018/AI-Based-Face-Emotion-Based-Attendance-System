import sys, cv2, json, csv, os
import numpy as np
from datetime import datetime
from PySide6.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget, QHBoxLayout
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtCore import QTimer, Qt, QDateTime
from tensorflow.keras.models import load_model

project_dir = r"D:\Python Projects\AI Based Face & Emotion Detection Based Attendance System"
face_model_path = os.path.join(project_dir, "face_recognition_model.keras")
emotion_model_path = os.path.join(project_dir, "face_emotion_model.keras")
face_label_path = os.path.join(project_dir, "class_labels.json")
emotion_label_path = os.path.join(project_dir, "emotion_labels.json")
csv_filename = os.path.join(project_dir, "attendance_log.csv")

face_model = load_model(face_model_path)
emotion_model = load_model(emotion_model_path)

with open(face_label_path, "r") as f:
    face_label_map = json.load(f)
rev_face_map = {v: k for k, v in face_label_map.items()}

with open(emotion_label_path, "r") as f:
    emotion_label_map = json.load(f)
rev_emotion_map = {v: k for k, v in emotion_label_map.items()}

if not os.path.exists(csv_filename):
    with open(csv_filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Name", "Date_Time", "Emotion", "PunchType"])


class AttendanceApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("💡 AI Face & Emotion Attendance System")
        self.setStyleSheet("""
            QWidget {
                background-color: #1e1e2f;
                color: #ffffff;
                font-family: 'Segoe UI';
                font-size: 16px;
            }
            QLabel#StatusLabel {
                background-color: #2a2a3d;
                padding: 12px;
                border-radius: 8px;
                font-size: 18px;
            }
            QPushButton {
                background-color: #0078d7;
                padding: 8px 20px;
                border-radius: 10px;
                color: white;
                font-size: 16px;
            }
            QPushButton:hover {
                background-color: #005999;
            }
        """)

        # UI Elements
        self.image_label = QLabel()
        self.image_label.setFixedSize(800, 500)
        self.image_label.setStyleSheet("border: 2px solid #444; border-radius: 10px;")
        self.image_label.setAlignment(Qt.AlignCenter)

        self.label_status = QLabel("👉 Please select IN or OUT to begin")
        self.label_status.setObjectName("StatusLabel")
        self.label_status.setAlignment(Qt.AlignCenter)

        self.clock_label = QLabel()
        self.clock_label.setStyleSheet("font-size: 14px; color: #ccc; padding: 5px;")

        self.button_in = QPushButton("🔵 IN")
        self.button_out = QPushButton("🔴 OUT")
        self.button_maximize = QPushButton("🔳 Maximize")

        self.button_in.clicked.connect(lambda: self.select_punch_type("IN"))
        self.button_out.clicked.connect(lambda: self.select_punch_type("OUT"))
        self.button_maximize.clicked.connect(self.toggle_window_mode)

        #Layouts
        top_layout = QHBoxLayout()
        top_label = QLabel("📌 AI Based Face & Emotion Based Attendance System")
        top_label.setStyleSheet("font-size: 20px; font-weight: bold; padding: 10px;")
        top_layout.addWidget(top_label)
        top_layout.addStretch()
        top_layout.addWidget(self.clock_label)
        top_layout.addSpacing(10)
        top_layout.addWidget(self.button_maximize)

        image_wrapper = QHBoxLayout()
        image_wrapper.addStretch()
        image_wrapper.addWidget(self.image_label)
        image_wrapper.addStretch()

        center_layout = QVBoxLayout()
        center_layout.addStretch()
        center_layout.addLayout(image_wrapper)
        center_layout.addSpacing(10)
        center_layout.addWidget(self.label_status)
        center_layout.addStretch()

        button_layout = QHBoxLayout()
        button_layout.addStretch()
        button_layout.addWidget(self.button_in)
        button_layout.addSpacing(20)
        button_layout.addWidget(self.button_out)
        button_layout.addStretch()

        main_layout = QVBoxLayout()
        main_layout.addLayout(top_layout)
        main_layout.addLayout(center_layout)
        main_layout.addSpacing(10)
        main_layout.addLayout(button_layout)
        main_layout.setContentsMargins(20, 20, 20, 20)
        self.setLayout(main_layout)

        # Variables
        self.punch_type = None
        self.last_name = "Unknown"
        self.last_emotion = "Unknown"
        self.ready_to_capture = False
        self.capture_done = False

        self.cap = cv2.VideoCapture(0)
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)

        self.clock_timer = QTimer()
        self.clock_timer.timeout.connect(self.update_clock)
        self.clock_timer.start(1000)
        self.update_clock()

    def update_clock(self):
        current = QDateTime.currentDateTime().toString("dd-MM-yyyy hh:mm:ss")
        self.clock_label.setText(f"🕒 {current}")

    def select_punch_type(self, punch_type):
        self.punch_type = punch_type
        self.ready_to_capture = True
        self.capture_done = False
        self.image_label.clear()
        self.label_status.setText(f"🟡 Punch type selected: {punch_type} — Look at the camera...")
        self.timer.start(30)

    def update_frame(self):
        ret, frame = self.cap.read()
        if not ret:
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

        self.last_name = "Unknown"
        self.last_emotion = "Unknown"
        MIN_FACE_WIDTH = 100

        for (x, y, w, h) in faces:
            if w >= MIN_FACE_WIDTH:
                face_rgb = rgb[y:y+h, x:x+w]
                face_rgb_resized = cv2.resize(face_rgb, (100, 100)) / 255.0
                face_input = np.expand_dims(face_rgb_resized, axis=0)
                pred = face_model.predict(face_input)[0]
                class_id = np.argmax(pred)
                confidence = pred[class_id]
                self.last_name = rev_face_map[class_id] if confidence > 0.8 else "Unknown"

                face_gray_resized = cv2.resize(face_rgb, (100, 100)) / 255.0
                emotion_input = np.expand_dims(face_gray_resized, axis=0)
                pred_emotion = emotion_model.predict(emotion_input)[0]
                emotion_id = np.argmax(pred_emotion)
                self.last_emotion = rev_emotion_map[emotion_id]

                label = f"{self.last_name} | {self.last_emotion}"
                cv2.rectangle(rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(rgb, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                if self.ready_to_capture and not self.capture_done and self.last_name != "Unknown":
                    current_time = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
                    self.save_or_update_attendance_csv(self.last_name, current_time, self.last_emotion, self.punch_type)
                    self.label_status.setText(f"✅ {self.punch_type} punch at {current_time} for {self.last_name}")
                    self.capture_done = True
                    self.punch_type = None
                    self.ready_to_capture = False
                    self.timer.stop()
                break

        image = QImage(rgb.data, rgb.shape[1], rgb.shape[0], QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(image))

    def save_or_update_attendance_csv(self, name, datetime_str, emotion, punch_type):
        date_str = datetime_str.split()[0]
        updated = False
        rows = []

        if os.path.exists(csv_filename):
            with open(csv_filename, mode='r', newline='') as file:
                reader = csv.reader(file)
                headers = next(reader)
                for row in reader:
                    if len(row) != 4:
                        continue
                    row_name, row_datetime, row_emotion, row_type = row
                    row_date = row_datetime.split()[0]

                    if row_name == name and row_type == punch_type and row_date == date_str:
                        old_time = datetime.strptime(row_datetime, "%d-%m-%Y %H:%M:%S")
                        new_time = datetime.strptime(datetime_str, "%d-%m-%Y %H:%M:%S")
                        if (punch_type == "IN" and new_time < old_time) or \
                           (punch_type == "OUT" and new_time > old_time):
                            row = [name, datetime_str, emotion, punch_type]
                        updated = True
                    rows.append(row)

        if not updated:
            rows.append([name, datetime_str, emotion, punch_type])

        with open(csv_filename, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["Name", "Date_Time", "Emotion", "PunchType"])
            writer.writerows(rows)

    def closeEvent(self, event):
        self.cap.release()
        event.accept()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.close()

    def toggle_window_mode(self):
        if self.isFullScreen():
            self.showNormal()
            self.button_maximize.setText("🔳 Maximize")
        else:
            self.showFullScreen()
            self.button_maximize.setText("🗗 Restore")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = AttendanceApp()
    win.showFullScreen()
    sys.exit(app.exec())
