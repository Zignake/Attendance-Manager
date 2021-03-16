import cv2
import os
import numpy as np

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

face_id = input('Enter the name of the person whose data is to be taken: ')
parent_dir = "dataset/"
path = os.path.join(parent_dir, face_id)
os.mkdir(path)

print("\n [INFO] Initializing face capture. Look the camera and wait ... ")

cap = cv2.VideoCapture(0)

count = 0
while(True):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=3,
        flags=cv2.CASCADE_SCALE_IMAGE,
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        count += 1

        cv2.imwrite(parent_dir + str(count) + ".jpg", gray[y:y+h, x:x+w])
        cv2.imshow('image', frame)

    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break
    elif count >= 30:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()
