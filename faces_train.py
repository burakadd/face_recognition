import os

import numpy
from PIL import Image
import cv2
import pickle


face_cascade = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_alt2.xml'
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'base')

y_labels = []
x_train = []
current_id = 0
label_ids = {}

for root, dirs, files in os.walk(IMAGE_DIR):
    for file in files:
        path = os.path.join(root, file)
        label = os.path.basename(
            (
                root.replace(' ', '_').lower()
             )
        )
        if not label in label_ids:
            label_ids[label] = current_id
            current_id += 1
        _id = label_ids[label]

        pil_image = Image.open(path).convert('L')

        size = 550, 550
        final_image = pil_image.resize(size, Image.ANTIALIAS)
        image_array = numpy.array(pil_image, 'uint8')
        faces = face_cascade.detectMultiScale(
            image_array,
            scaleFactor=1.5,
            minNeighbors=5,
        )

        for x, y, w, h in faces:
            roi = image_array[y: y + h, x: x + w]
            x_train.append(roi)
            y_labels.append(_id)

with open('labels.pickle', 'wb') as f:
    pickle.dump(label_ids, f)

recognizer.train(x_train, numpy.array(y_labels))
recognizer.save('trainner.yml')

