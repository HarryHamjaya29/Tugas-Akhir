import cv2
import sys
import logging as log
import datetime as dt
import time
from sklearn.externals import joblib    
from IPython.display import display, Image
import pandas as pd
import math
import os
import numpy as np
import face_recognition
import imageio
from mlxtend.image import extract_face_landmarks

import warnings
warnings.filterwarnings("ignore")

cascPath = "haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascPath)
log.basicConfig(filename='webcam.log',level=log.INFO)

video_capture = cv2.VideoCapture(0)
anterior = 0
checked = 0

# weight_model = 'weight_predictor_embedding.model'
weight_model = 'weight_predictor_weight_best.model'
# height_model = 'height_predictor_embedding.model'
# height_model = 'height_predictor_height_best.model'
height_model = 'new_height_predictor_height_best.model'
# bmi_model = 'bmi_predictor_embedding.model'
bmi_model = 'bmi_predictor_bmi_best.model'

height_model = joblib.load(height_model)
weight_model = joblib.load(weight_model)
bmi_model = joblib.load(bmi_model)

def get_face_encoding(image_path):
    picture_of_me = image_path
    my_face_encoding = face_recognition.face_encodings(picture_of_me)
    if not my_face_encoding:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        print("no face found !!!",current_time)
        return np.zeros(128).tolist()
    return my_face_encoding[0].tolist()

def get_face_encoding_last(image_path):
    # print(image_path)
    # img = cv2.imread(image_path)
    # img = face_recognition.load_image_file(image_path)
    l = face_recognition.face_landmarks(image_path)
    
    landmarks = []
    for d in l:
        for key, val in d.items():
            if val is None:
                continue
            landmarks.extend(val)

    if len(landmarks)==0:
        return 255, 0
    miniy = 10000009
    maxiy = -1
    vertical = 0
    minix = 10000009
    maxix = -1
    horizontal = 0
    for i in range(len(landmarks)):
        if landmarks[i][1]<miniy:
            miniy = landmarks[i][1]
        if landmarks[i][1]>maxiy:
            maxiy = landmarks[i][1]
        if landmarks[i][0]<minix:
            minix = landmarks[i][0]
        if landmarks[i][0]>maxix:
            maxix = landmarks[i][0]
    vertical = maxiy-miniy        
    horizontal = maxix-minix
    temp = 0
#     if vertical>100:
#         temp = vertical//100
#         vertical = vertical - (temp-1)*100 - random.randrange(25, 40)
    # print([vertical,horizontal])
    

#         for j in range(68):
#             if i<j:
#                 distances.append(math.sqrt((landmarks[i][0] - landmarks[j][0]) ** 2 + (landmarks[i][1] - landmarks[j][1]) ** 2))
    distances = [vertical,horizontal]
    return vertical,horizontal

def predict_height_width_BMI(test_image,height_model,weight_model,bmi_model):
    test_array = np.expand_dims(np.array(get_face_encoding(test_image)),axis=0)
    # height = np.asscalar(np.exp(height_model.predict(test_array)))
    # weight = np.asscalar(np.exp(weight_model.predict(test_array)))
    # bmi = np.asscalar(np.exp(bmi_model.predict(test_array)))
    length,width = get_face_encoding_last(test_image)
    temp_height = [[0, length]]
    height = math.exp(height_model.predict(temp_height))
    temp_width = [[0, height]]
    weight = math.exp(weight_model.predict(temp_width))
    weight = round(weight, 2)
    temp_bmi = [[height,weight]]
    bmi = math.exp(bmi_model.predict(temp_bmi))
    bmi = round(bmi, 2)
    return height, weight, bmi, length, width

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
    weight_org = (10, 40)
    bmi_org = (10, 55) 
    height_err = (10, 70)
    weight_err = (10, 85)
    bmi_err = (10, 100) 
    
    # fontScale 
    fontScale = 0.6
    
    # Blue color in BGR 
    color = (255, 0, 0) 
    
    # Line thickness of 2 px 
    thickness = 1
    
    if checked%31==0:
        height_value, weight_value, bmi_value, length, width = predict_height_width_BMI(frame,height_model,weight_model,bmi_model)

    
    checked+=1
    
    abs_error_height = round(abs((height_value-161)/161)*100,2)
    abs_error_weight = round(abs((weight_value-60.5)/60.5)*100,2)
    abs_error_bmi = round(abs((bmi_value-24.5)/24.5)*100,2)

    # Using cv2.putText() method 
    frame = cv2.putText(frame, "Height = " + str(height_value), height_org, font,  fontScale, color, thickness, cv2.LINE_AA)
    frame = cv2.putText(frame, "Weight = " + str(weight_value), weight_org, font,  fontScale, color, thickness, cv2.LINE_AA)
    frame = cv2.putText(frame, "BMI = " + str(bmi_value), bmi_org, font,  fontScale, color, thickness, cv2.LINE_AA)
    frame = cv2.putText(frame, "Error Height = " + str(abs_error_height)+ "%", height_err, font,  fontScale, color, thickness, cv2.LINE_AA)
    frame = cv2.putText(frame, "Error Weight = " + str(abs_error_weight)+ "%", weight_err, font,  fontScale, color, thickness, cv2.LINE_AA)
    frame = cv2.putText(frame, "Error BMI = " + str(abs_error_bmi)+ "%", bmi_err, font,  fontScale, color, thickness, cv2.LINE_AA)


    # Display the resulting frame
    cv2.imshow('Video', frame)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # Display the resulting frame
    cv2.imshow('Video', frame)

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
