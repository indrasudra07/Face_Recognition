import numpy as np
import cv2

captureDevice = cv2.VideoCapture(0)  # captureDevice = camera

faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = captureDevice.read()
    faces = faceDetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        x1, y1 = x+w, y+h
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 255), 1)

        cv2.line(frame, (x, y), (x+30, y), (255, 0, 255), 6)  # Top Left
        cv2.line(frame, (x, y), (x, y+30), (255, 0, 255), 6)

        cv2.line(frame, (x1, y), (x1-30, y), (255, 0, 255), 6)  # Top Right
        cv2.line(frame, (x1, y), (x1, y+30), (255, 0, 255), 6)

        cv2.line(frame, (x, y1), (x+30, y1), (255, 0, 255), 6)  # Bottom Left
        cv2.line(frame, (x, y1), (x, y1-30), (255, 0, 255), 6)

        cv2.line(frame, (x1, y1), (x1-30, y1),
                 (255, 0, 255), 6)  # Bottom Right
        cv2.line(frame, (x1, y1), (x1, y1-30), (255, 0, 255), 6)

    cv2.imshow('my frame', frame)
    if cv2.waitKey(1) == ord('q'):
        break

captureDevice.release()
cv2.destroyAllWindows()
