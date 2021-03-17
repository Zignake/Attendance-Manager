import cv2
import numpy as np
from PIL import Image
import os

personListFile = open("personList.txt", "r")
personList = personListFile.readlines()
for i, person in enumerate(personList):
    personList[i] = person[:-1]
personListFile.close()

path = 'dataset/'
recognizer = cv2.face.LBPHFaceRecognizer_create()


def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faceSamples = []
    ids = []

    for imagePath in imagePaths:
        val = imagePath.split('/')[1]
        currentId = personList.index(val)
        for i in range(30):
            img = cv2.imread(imagePath + '/' + str(i+1) + '.jpg', 0)
            faceSamples.append(img)
            ids.append(currentId)

    return faceSamples, ids


print("\n [INFO] Training faces. It will take a few seconds. Wait ...")

faces, ids = getImagesAndLabels(path)
recognizer.train(faces, np.array(ids))

# Save the model into trainer/trainer.yml
recognizer.write('trainer/trainer.yml')

# Print the numer of faces trained and end program
print("\n [INFO] {0} faces trained. Exiting Program".format(
    len(np.unique(ids))))
