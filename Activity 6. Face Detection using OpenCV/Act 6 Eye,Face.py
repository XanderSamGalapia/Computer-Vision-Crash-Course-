import cv2
def detect():
    face_cascade = cv2.CascadeClassifier('D:/Elective3/haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier('D:/Elective3/haarcascade_eye.xml')
    camera = cv2.VideoCapture(0)

    while True:
        ret, frame = camera.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
            roi_gray = gray[y:y + h, x:x + w]
            roi_color = frame[y:y + h, x:x + w]

            eyes = eye_cascade.detectMultiScale(roi_gray, 1.03, 10, 0, (40, 40))
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (0, 50, 100), 3)

        cv2.imshow("Camera", frame)
        if cv2.waitKey(1000 // 12) & 0xFF == ord("q"): 
            break

    camera.release()
    cv2.destroyAllWindows()
    
cv2.waitKey(0)
if __name__ == "__main__":
    detect()
