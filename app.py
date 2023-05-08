import cv2

# Load the pre-trained model for face detection
face_cascade = cv2.CascadeClassifier("cascades/haarcascade_frontalface_default.xml")

# Load the pre-trained model for face recognition
recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trained_data.xml")

# Initialize the video capture object
cap = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)

    # Iterate over each detected face
    for x, y, w, h in faces:
        # Draw a rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Extract the face region from the frame
        face_roi = gray[y : y + h, x : x + w]

        # Recognize the face using the pre-trained model
        label, confidence = recognizer.predict(face_roi)

        # Print the predicted label and confidence
        print("Label:", label, "Confidence:", confidence)

    # Display the resulting frame
    cv2.imshow("frame", frame)

    # Exit if the 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the capture object and close all windows
cap.release()
cv2.destroyAllWindows()
