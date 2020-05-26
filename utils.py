import cv2

face_cascade = cv2.CascadeClassifier(
    'cascades/haarcascade_frontalface_alt2.xml'
)

cap = cv2.VideoCapture(0)

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



    cv2.imshow('myframe', frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
