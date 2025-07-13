import pickle
import mediapipe as mp
import cv2
import numpy as np
import getResult

mp_drawing = mp.solutions.drawing_utils #drawing functions
mp_holistic = mp.solutions.holistic #solutions

with open('training/LR.pkl', 'rb') as f:
    model = pickle.load(f)

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
                                    mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1))
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

       
       
        #write in csv by calling functions
        try:
            res = getResult.getResult(results.pose_landmarks.landmark,results.face_landmarks.landmark, model)
            coords = tuple(np.multiply(
                            np.array(
                                (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].x, 
                                 results.pose_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_EAR].y))
                        , [640,480]).astype(int))
            cv2.rectangle(image, 
                          (coords[0], coords[1]-65), 
                          (coords[0]+len(res)*20, coords[1]-100), 
                          (245, 117, 16), -1)
            cv2.putText(image, res, (coords[0], coords[1]-75), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        except:
            pass
        cv2.imshow('Raw Webcam Feed', image)
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
