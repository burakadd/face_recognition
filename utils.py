import cv2
import pickle

face_cascade = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_alt2.xml'
)
eye_cascade = cv2.CascadeClassifier(
    'cascades/haarcascade_eye.xml'
)
smile_cascade = cv2.CascadeClassifier(
    'cascades/haarcascade_eye.xml'
)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read('trainner.yml')

cap = cv2.VideoCapture(0)
with open('labels.pickle', 'rb') as f:
    og_labels = pickle.load(f)
    labels = {v: k for k, v in og_labels.items()}

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.5,
        minNeighbors=5
    )

    for x, y, w, h in faces:
        print(f'x: {x}, y: {y}, w: {w}, h: {h}')

        roi_gray = gray[y: y + h, x: x + w]
        roi_colour = frame[y: y + h, x: x + w]
        img_item = 'my-image.png'
        cv2.imwrite(img_item, roi_gray)

        _id, conf = recognizer.predict(roi_gray)

        if conf >= 45:
            font = cv2.FONT_HERSHEY_SIMPLEX
            name = labels[_id]
            color = 255, 255, 255
            stroke = 2
            cv2.putText(
                frame,
                name,
                (x, y),
                font,
                1,
                color,
                stroke,
                cv2.LINE_AA,
            )

        color = 255, 0, 0
        stroke = 2
        width, height = x + w, y + h
        cv2.rectangle(
            img=frame,
            pt1=(x, y),
            pt2=(width, height),
            color=color,
            thickness=stroke,
        )

    cv2.imshow('myframe', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
