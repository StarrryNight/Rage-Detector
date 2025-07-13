import mediapipe as mp
import cv2
import csv
import os
import numpy as np

def initializeCSV(pose_landmarks,face_landmarks):
    landmarks = ['class']
    for val in range(1, len(face_landmarks)+len(pose_landmarks)+1):
        landmarks += [f'x{val}', f'y{val}', f'z{val}', f'v{val}']

    #initialize csv
    with open('coords.csv', mode='w', newline='') as f:
        csv_writer = csv.writer(f,delimiter=',', quotechar ='"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(landmarks)


def exportCSV(pose_landmarks,face_landmarks, feeling):



     #get coordinates
    #get row
    pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose_landmarks]).flatten())
    face_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in face_landmarks]).flatten())
    row = pose_row+face_row
    row.insert(0,feeling)       
    print(feeling)

    #export to csv
    with open('coords.csv', mode = 'a', newline='') as f:
        csv_writer = csv.writer(f, delimiter = ',', quotechar='"', quoting = csv.QUOTE_MINIMAL)
        csv_writer.writerow(row)



    