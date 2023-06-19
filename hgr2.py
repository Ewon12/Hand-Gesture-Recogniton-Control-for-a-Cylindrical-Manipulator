import cv2
import numpy as np
import pickle
import time
import serial

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands
arduinoData=serial.Serial('COM5',9600)

class mpHands:
    import mediapipe as mp
    def __init__(self,maxHands=2,tol1=.5,tol2=.5):
        self.hands=self.mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=maxHands,min_detection_confidence=tol1, min_tracking_confidence=tol2)
    def Marks(self,frame):
        myHands=[]
        handsType=[]
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
            for hand in results.multi_handedness:
                handType=hand.classification[0].label
                handsType.append(handType)
            
            for handLandMarks in results.multi_hand_landmarks:
                myHand=[]
                for LandMark in handLandMarks.landmark:
                    myHand.append((int(LandMark.x*width),int(LandMark.y*height)))
                myHands.append(myHand)
        return myHands,handsType


def findDistances(handData):
    distMatrix= np.zeros([len(handData),len(handData)],dtype='float')
    palmSize=((handData[0][0]-handData[9][0])**2+(handData[0][1]-handData[9][1])**2)**(1./2.)
    for row in range(0,len(handData)):
        for column in range(0,len(handData)):
            distMatrix[row][column]=(((handData[row][0]-handData[column][0])**2+(handData[row][1]-handData[column][1])**2)**(1./2.))/palmSize
    return distMatrix

def findError(gestureMatrix,unknownMatrix,Keypoints):
    error=0
    for row in Keypoints:
        for column in Keypoints:
            error=error+abs(gestureMatrix[row][column]-unknownMatrix[row][column])
    return error

def findGesture(unknownGesture,knownGestures,Keypoints,gestNames,tol):
    errorArray=[]
    for i in range(0,len(gestNames),1):
        error=findError(knownGestures[i],unknownGesture,Keypoints)
        errorArray.append(error)
    errorMin=errorArray[0]
    minIndex=0
    for i in range(0,len(errorArray),1):
        if errorArray[i]<errorMin:
            errorMin=errorArray[i]
            minIndex=i
    if errorMin<tol:
        gesture=gestNames[minIndex]
    if errorMin>=tol:
        gesture=''
    return gesture


width=1200
height=720
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
findHands = mpHands(2)
time.sleep(1)
Keypoints = [0,4,5,8,9,12,13,16,17,20]

train=int(input('Enter 1 to Train, Enter 0 to Recognize'))

if train==1:
    trainCnt=0
    knownGestures=[]
    numGest= 9
    gestNames=['UP', 'DOWN', 'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD', 'GRIP', 'UNGRIP', 'RESET']
    trainName=input('Filename for training data? (Press Enter for Default)')
    if trainName=='':
        trainName='default'
    trainName=trainName+'.pkl'
    
if train==0:
    trainName=input('What Training Data Do you Want to Use? (Press Enter for Default)')
    if trainName=='':
        trainName='default'
    trainName=trainName+'.pkl'
    with open(trainName,'rb') as f:
        gestNames=pickle.load(f)
        knownGestures=pickle.load(f)

tol=50
cTime=0
pTime=0


while True:
    ignore, frame = cam.read()
    frame = cv2.resize(frame,(width,height))
    frame = cv2.flip(frame,1)
    
    start_point = (700, 200)
    end_point = (1100, 600)
    color = (0, 255, 0) 
    thickness = 2
    cv2.rectangle(frame, start_point, end_point, color, thickness)
    
    handData,handsType=findHands.Marks(frame)
    
    if train == 1:
        if handData !=[]:
            print('Please Show Gesture',gestNames[trainCnt],': Press t when Ready')
            if cv2.waitKey(1)  & 0xff == ord('t'):
                knownGesture=findDistances(handData[0])
                knownGestures.append(knownGesture)
                trainCnt=trainCnt+1
                if trainCnt==numGest:
                    train=0
                    with open(trainName,'wb') as f:
                        pickle.dump(gestNames,f)
                        pickle.dump(knownGestures,f)
    if train == 0:
        if handData !=[]:
            unknownGesture=findDistances(handData[0])
            myGesture=findGesture(unknownGesture,knownGestures,Keypoints,gestNames,tol)
            cv2.putText(frame,myGesture,(600,150),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),8)
            myGesture=myGesture+'\r'
            arduinoData.write(myGesture.encode())
    

    cv2.imshow('HGR', frame)
    cv2.moveWindow('HGR',0,0)
    key = cv2.waitKey(1)
    if key == ord('q'):
     break  

    cam.release()
