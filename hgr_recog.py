import cv2 as cv
import mediapipe as mp
import pandas as pd  
import numpy as np
import pickle
import serial

arduinoData=serial.Serial('COM5',9600)
width=720
height=480
cam=cv.VideoCapture(1,cv.CAP_DSHOW)
cam.set(cv.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv.CAP_PROP_FPS, 60)
cam.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc(*'MJPG'))
if not cam.isOpened():
    print("Cannot open camera")
    exit()
i = 0 


def image_processed(hand_img):

    img_rgb = cv.cvtColor(hand_img, cv.COLOR_BGR2RGB)
    img_flip = cv.flip(img_rgb, 1)
    mp_hands = mp.solutions.hands
    mp_Draw=mp.solutions.drawing_utils
    hands = mp_hands.Hands(static_image_mode=True,
    max_num_hands=1, min_detection_confidence=0.7)
    output = hands.process(img_flip)
    hands.close()

    try:
        data = output.multi_hand_landmarks[0]
        data = str(data)

        data = data.strip().split('\n')

        garbage = ['landmark {', '  visibility: 0.0', '  presence: 0.0', '}']

        without_garbage = []

        for i in data:
            if i not in garbage:
                without_garbage.append(i)
                        
        clean = []

        for i in without_garbage:
            i = i.strip()
            clean.append(i[2:])

        for i in range(0, len(clean)):
            clean[i] = float(clean[i])
        return(clean)
    except:
        return(np.zeros([1,63], dtype=int)[0])


with open('model.pkl', 'rb') as f:
    svm = pickle.load(f)



while True:
    
    ret, frame = cam.read()
    frame = cv.resize(frame,(width,height))
    
    start_point = (100, 350)
    end_point = (400, 100)
    color = (0, 255, 0) 
    thickness = 2
    cv.rectangle(frame, start_point, end_point, color, thickness)
    
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    frame = cv.flip(frame,1)
    data = image_processed(frame)
    
    data = np.array(data)
    y_pred = svm.predict(data.reshape(-1,63))
    font = cv.FONT_HERSHEY_SIMPLEX
    org = (50, 100)
    fontScale = 3
    color = (255, 0, 0)
    thickness = 5
    frame = cv.putText(frame, str(y_pred[0]), org, font, fontScale, color, thickness, cv.LINE_AA)
    y_pred = str(y_pred[0])+'\r'
    arduinoData.write(y_pred.encode())
    cv.imshow('frame', frame)
    
        
    if cv.waitKey(1) == ord('q'):
        break

cam.release()
cv.destroyAllWindows()