from Read import read_images
import numpy as np
import cv2

def face_rec():
    names = ['jm', 'xander']  # Put your names here for faces to recognize

    [X, y] = read_images("dataset", sz=(200, 200))
    y = np.asarray(y, dtype=np.int32)

    model = cv2.face.EigenFaceRecognizer_create()
    model.train(X, y)

    camera = cv2.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier('D:/Elective3/Activity 7/dataset/haarcascade_frontalface_default.xml')

    while True:
        ret, img = camera.read()
        if not ret:
            break

        faces = face_cascade.detectMultiScale(img, 1.3, 5)

        for (x, y, w, h) in faces:
            gray = cv2.cvtColor(img[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY)
            roi = cv2.resize(gray, (200, 200), interpolation=cv2.INTER_LINEAR)

            try:
                params = model.predict(roi)
                label = names[params[0]]
                confidence = params[1]  

                if confidence < 10: 
                    color = (0, 255, 0) # Green color for recognized faces
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, label + ", " + str(confidence), (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
                else:
                    color = (0, 0, 255) #Red color for unrecognised faces
                    cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                    cv2.putText(img, "Unrecognized", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

            except Exception as e:
                print(f"Error during prediction: {e}")
                continue

        cv2.imshow("camera", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    face_rec()