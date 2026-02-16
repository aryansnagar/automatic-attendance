import cv2
import os
import numpy as np
import time
from datetime import datetime

path = 'students'
recognizer = cv2.face.LBPHFaceRecognizer_create()
detector = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def get_images_and_labels(path):
    image_paths = [os.path.join(path, f) for f in os.listdir(path)]
    face_samples = []
    ids = []
    name_map = {}

    for i, image_path in enumerate(image_paths):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        name = os.path.split(image_path)[-1].split(".")[0]
        
        faces = detector.detectMultiScale(img)
        for (x, y, w, h) in faces:
            face_samples.append(img[y:y+h, x:x+w])
            ids.append(i)
            name_map[i] = name
    return face_samples, ids, name_map

print("Training recognizer...")
faces, ids, name_mapping = get_images_and_labels(path)
recognizer.train(faces, np.array(ids))

def markAttendance(name):
    with open('attendance.txt', 'a+') as f:
        f.seek(0)
        lines = f.readlines()
        names_in_file = [line.split(',')[0] for line in lines]
        if name not in names_in_file:
            now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(f"{name}, {now}\n")

cap = cv2.VideoCapture(0)
last_check = time.time()

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if time.time() - last_check > 2:
        faces = detector.detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            id_num, confidence = recognizer.predict(gray[y:y+h, x:x+w])

            if confidence < 70:
                name = name_mapping[id_num]
                markAttendance(name)
                print(f"Matched: {name} ({round(100 - confidence)}% match)")
        
        last_check = time.time()

    cv2.imshow('Attendance System', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
