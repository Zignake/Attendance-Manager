import cv2
import numpy as np
import os

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainer/trainer.yml')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX

personListFile = open("personList.txt", "r")
personList = personListFile.readlines()
for person in personList:
    person = person[:-1]
personListFile.close()

cap = cv2.VideoCapture(0)
minW = 0.1*cap.get(3)
minH = 0.1*cap.get(4)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH),),
    )
    for(x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        predictedId, confidence = recognizer.predict(gray[y:y+h, x:x+w])

        if (confidence < 100):
            predictedFace = personList[predictedId]
            confidence = "  {0}%".format(round(100 - confidence))
        else:
            predictedFace = "unknown"
            confidence = "  {0}%".format(round(100 - confidence))

        cv2.putText(
            img,
            str(predictedFace),
            (x+5, y-5),
            font,
            1,
            (255, 255, 255),
            2
        )
        cv2.putText(
            img,
            str(confidence),
            (x+5, y+h-5),
            font,
            1,
            (255, 255, 0),
            1
        )

    cv2.imshow('camera', img)
    k = cv2.waitKey(1) & 0xff  #
    if k == 27:
        break

print("\n [INFO] Exiting Program and cleanup stuff")
cap.release()
cv2.destroyAllWindows()
