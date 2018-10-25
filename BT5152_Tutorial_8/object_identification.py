import os
import numpy as np
import cv2 as cv
from keras.preprocessing import image
from keras.applications.resnet50 import ResNet50
from keras.applications.resnet50 import preprocess_input, decode_predictions

if __name__ == "__main__":
    # ResNet50 image size requirements
    img_h, img_w = 224, 224

    # Load ResNet50 with weights from imagenet
    model = ResNet50(weights='imagenet')

    # Set camera resolution
    resolution = (640, 480)

    # Start video capture
    cap = cv.VideoCapture(0)
    cap.set(4, resolution[0])
    cap.set(3, resolution[1])

    # Offset for the text location
    text_location = (int(resolution[0] / 5), int(resolution[1] * 2.5 / 3))

    while True:
        # Begin reading the data from the video capture
        ret, frame = cap.read()
        # Flip the image because the camera is looking at us in a different direction
        frame = cv.flip(frame, +1)

        # Downsample the video frame to the model requirements
        X = cv.resize(frame, (img_h, img_w), interpolation=cv.INTER_AREA)
        X = np.expand_dims(X, axis=0)
        X = preprocess_input(X)

        # Use the model to make predictions
        preds = model.predict(X)

        # Write the predictions on the frame
        for i, pred in enumerate(decode_predictions(preds, top=3)[0]):
            text = "{}: {:.3f}".format(pred[1], pred[2])
            x, y = text_location[0], text_location[1] + i * 40
            cv.putText(frame, text, (x, y), cv.FONT_HERSHEY_DUPLEX, 1.4, (0, 255, 0), 2)

        # Show the frames as a video
        cv.imshow('video', frame)

        # Press q to stop the program
        if cv.waitKey(20) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()
