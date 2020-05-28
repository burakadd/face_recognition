import os
from typing import Tuple

import numpy
import cv2
import pickle


face_cascade = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_alt2.xml'
)
recognizer = cv2.face.LBPHFaceRecognizer_create()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'base')


def resize_image(size: Tuple[int, int], path_to_image):
    from PIL import Image
    image = Image.open(path_to_image).convert('L')
    return image.resize(
        size, Image.ANTIALIAS
    )


def train_recognizer(image_directory, recognizer):
    labels_for_recognized_faces: list = []
    region_of_interests: list = []
    current_id: int = 0
    label_ids: dict = {}
    for root, dirs, files in os.walk(image_directory):
        for file in files:
            path: str = os.path.join(root, file)
            label: str = os.path.basename(
                (
                    root.replace(' ', '_').lower()
                 )
            )
            if label not in label_ids:
                label_ids[label] = current_id
                current_id += 1
            _id = label_ids[label]

            image = resize_image((550, 550), path)

            image_array = numpy.array(image, 'uint8')
            faces = face_cascade.detectMultiScale(
                image_array,
                scaleFactor=1.5,
                minNeighbors=5,
            )

            for x, y, w, h in faces:
                region_of_interests = image_array[y: y + h, x: x + w]
                region_of_interests.append(region_of_interests)
                labels_for_recognized_faces.append(_id)

    with open('labels.pickle', 'wb') as f:
        pickle.dump(label_ids, f)

    recognizer.train(region_of_interests, numpy.array(labels_for_recognized_faces))
    recognizer.save('trainner.yml')

