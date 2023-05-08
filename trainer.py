import cv2
import os
import numpy as np

# Define paths
base_dir = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(base_dir, "images")

# Load face detection cascade
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# Initialize lists to store face and label data
faces = []
labels = []

# Loop through image directory and read images
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith("jpg") or file.endswith("png"):
            path = os.path.join(root, file)
            label = os.path.basename(os.path.dirname(path))
            img = cv2.imread(path)

            # Convert image to grayscale
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Detect faces in the image
            faces_rects = face_cascade.detectMultiScale(gray, 1.3, 5)

            # Add faces and labels to lists
            for x, y, w, h in faces_rects:
                faces.append(gray[y : y + h, x : x + w])
                labels.append(int(label))

# Train face recognition model
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.train(faces, np.array(labels))

# Save trained model to file
recognizer.save(os.path.join(base_dir, "trained_data.xml"))
