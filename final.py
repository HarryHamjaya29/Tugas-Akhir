import cv2
import sys
import logging as log
import datetime as dt
from time import sleep
from sklearn.externals import joblib    
from IPython.display import display, Image
import pandas as pd
import os
import numpy as np
import face_recognition

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
#count = 0

weight_model = 'weight_predictor_embedding.model'
height_model = 'height_predictor_embedding.model'
bmi_model = 'bmi_predictor_embedding.model'

height_model = joblib.load(height_model)
weight_model = joblib.load(weight_model)
bmi_model = joblib.load(bmi_model)

def get_face_encoding(image_path):
    #print(image_path)
    #picture_of_me = face_recognition.load_image_file(image_path)
    picture_of_me = image_path
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        print("no face found !!!")
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()

def predict_height_width_BMI(test_image,height_model,weight_model,bmi_model):
    test_array = np.expand_dims(np.array(get_face_encoding(test_image)),axis=0)
    height = np.asscalar(np.exp(height_model.predict(test_array)))
    weight = np.asscalar(np.exp(weight_model.predict(test_array)))
    bmi = np.asscalar(np.exp(bmi_model.predict(test_array)))
    return height, weight, bmi

while True:
    if not video_capture.isOpened():
        print('Unable to load camera.')
        sleep(5)
        pass

    # Capture frame-by-frame
    ret, frame = video_capture.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30)
    )

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    if anterior != len(faces):
        anterior = len(faces)
        log.info("faces: "+str(len(faces))+" at "+str(dt.datetime.now()))
    
    # font 
    font = cv2.FONT_HERSHEY_SIMPLEX 
    
    # org 
    height_org = (10, 25)
    weight_org = (10, 45)
    bmi_org = (10, 65) 
    
    # fontScale 
    fontScale = 0.7
    
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 2

    #path = '../Test Images/img_4183.bmp'

    height_value, weight_value, bmi_value = predict_height_width_BMI(frame,height_model,weight_model,bmi_model)
    
    # Using cv2.putText() method 
    frame = cv2.putText(frame, "Height = " + str(height_value), height_org, font,  fontScale, color, thickness, cv2.LINE_AA)
    frame = cv2.putText(frame, "Weight = " + str(weight_value), weight_org, font,  fontScale, color, thickness, cv2.LINE_AA)
    frame = cv2.putText(frame, "BMI = " + str(bmi_value), bmi_org, font,  fontScale, color, thickness, cv2.LINE_AA)

    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
