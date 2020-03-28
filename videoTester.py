import os
import cv2 as cv
#print(cv.__file__)
import numpy as np
from keras.models import model_from_json
from keras.preprocessing import image

#load model
model = model_from_json(open("fer.json", "r").read())
#load weights
model.load_weights('fer.h5')


face_haar_cascade = cv.CascadeClassifier('D:\ProgramData\lib\site-packages\cv2\data\haarcascade_frontalface_default.xml')


cap=cv.VideoCapture(0)

while True:
    ret,test_img=cap.read()# captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv.cvtColor(test_img, cv.COLOR_BGR2GRAY)
    gray_img = np.array(gray_img, dtype='uint8')
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.3, 5)


    for (x,y,w,h) in faces_detected:
        cv.rectangle(test_img,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        roi_gray=gray_img[y:y+w,x:x+h]#cropping region of interest i.e. face area from  image
        roi_gray=cv.resize(roi_gray,(48,48))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis = 0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        #find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

    resized_img = cv.resize(test_img, (1000, 700))
    cv.imshow('Facial emotion analysis ',resized_img)



    if cv.waitKey(10) == ord('q'):#wait until 'q' key is pressed
        break

cap.release()
cv.destroyAllWindows