import mediapipe as mp
import cv2
import numpy as np
import handleCSV

mp_drawing = mp.solutions.drawing_utils #drawing functions
mp_holistic = mp.solutions.holistic #solutions

current_feeling = "angry" #current training feeling


csv_initialized =  True #If False, restarts training. If true, appends on current data


cap = cv2.VideoCapture(0)


with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:

    while cap.isOpened():
        ret, frame = cap.read()
        #Recolor
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        

        #Face
        
        results = holistic.process(image)
        image = cv2.cvtColor(image,cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(image,results.face_landmarks, mp_holistic.FACEMESH_CONTOURS, 
                                    mp_drawing.DrawingSpec(color=(80,110,10), thickness =1, circle_radius=1),
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=2, circle_radius=2))
        #Hands
        mp_drawing.draw_landmarks(image,results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(80,22,10), thickness =1, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2))
        mp_drawing.draw_landmarks(image,results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(121,22,76), thickness =1, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2))
        #Pose
        mp_drawing.draw_landmarks(image,results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                    mp_drawing.DrawingSpec(color=(245,117,66), thickness =1, circle_radius=4),
                                    mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))

        cv2.imshow('Raw Webcam Feed', image)
        
        #write in csv by calling functions
        try:
            if not csv_initialized:
                handleCSV.initializeCSV(results.pose_landmarks.landmark,results.face_landmarks.landmark)
                csv_initialized= True
            print("hio")
            handleCSV.exportCSV(results.pose_landmarks.landmark,results.face_landmarks.landmark,current_feeling)
        except:
            pass

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()