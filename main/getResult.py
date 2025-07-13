import mediapipe as mp
import cv2
import csv
import os
import numpy as np
import pandas as pd


def getResult(pose_landmarks,face_landmarks, model)-> str:

    #get coordinates
    #get row
    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten())
    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face_landmarks]).flatten())
    row = pose_row+face_row

    X = pd.DataFrame([row])
    body_language = model.predict(X.values)[0]
    prob = model.predict_proba(X.values)[0]
    print(body_language, prob)
    return body_language

    



    