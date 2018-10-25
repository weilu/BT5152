import os
import numpy as np
import cv2 as cv

if __name__ == "__main__":
    # Load the Haar Cascade models, you will need to download the models
    # Refer to README.md for instructions
    CASCADE_DIR = os.path.join(os.getcwd(), 'haarcascades')
    FRONTAL_FACE = os.path.join(CASCADE_DIR, 'haarcascade_frontalface_default.xml')
    EYE = os.path.join(CASCADE_DIR, 'haarcascade_eye.xml')

    face_cascade = cv.CascadeClassifier(FRONTAL_FACE)
    eye_cascade = cv.CascadeClassifier(EYE)

    # Set camera resolution
    resolution = (640, 480)

    # Start video capture
    cap = cv.VideoCapture(0)
    cap.set(4, resolution[0])
    cap.set(3, resolution[1])

    while True:
        # Begin reading the data from the video capture
        ret, frame = cap.read()
        # Flip the image because the camera is looking at us in a different direction
        frame = cv.flip(frame, +1)

        # Transform the colour channels to grayscale as the model can only handle
        # 1 channel
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

        # Using the face detector to find faces
        faces = face_cascade.detectMultiScale(gray,
                                              scaleFactor=1.3,
                                              minSize=(24, 24))

        # For each face found in the frame, draw a rectangle
        for (x, y, w, h) in faces:
            cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray,
                                                scaleFactor=1.3,
                                                minSize=(20, 20))

            for (ex, ey, ew, eh) in eyes:
                cv.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

        # Show the frames as a video
        cv.imshow('video', frame)

        # Press q to stop the program
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
