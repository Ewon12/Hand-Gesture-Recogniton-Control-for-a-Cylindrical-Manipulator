import cv2
import numpy as np
import pickle
import time
import serial


#arduinoData=serial.Serial('COM5',9600)

class mpHands:
    import mediapipe as mp
    def __init__(self,maxHands=2,tol1=.5,tol2=.5):
        self.hands=self.mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=maxHands,min_detection_confidence=tol1, min_tracking_confidence=tol2)
        self.mpDraw=self.mp.solutions.drawing_utils
    def Marks(self,frame):
        myHands=[]
        frameRGB=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        results=self.hands.process(frameRGB)
        if results.multi_hand_landmarks != None:
                for handLandMarks in results.multi_hand_landmarks:
                    myHand=[]
                    self.mpDraw.draw_landmarks(frame,handLandMarks,self.mp.solutions.hands.HAND_CONNECTIONS)
                    for LandMark in handLandMarks.landmark:
                        myHand.append((int(LandMark.x*width),int(LandMark.y*height)))
                    myHands.append(myHand)

        return myHands

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


width=920
height=640
cam=cv2.VideoCapture(0,cv2.CAP_DSHOW)
cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cam.set(cv2.CAP_PROP_FRAME_HEIGHT,height)
cam.set(cv2.CAP_PROP_FPS, 30)
cam.set(cv2.CAP_PROP_FOURCC,cv2.VideoWriter_fourcc(*'MJPG'))
findHands = mpHands(1)
time.sleep(1)
Keypoints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]

train=int(input('Enter 1 to Train, Enter 0 to Recognize'))

if train==1:
    trainCnt=0
    knownGestures=[]
    numGest= 10
    gestNames=['UP', 'DOWN', 'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD', 'GRIP', 'UNGRIP', 'RESET','Go Auto']
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


while True:
    ignore, frame = cam.read()
    frame = cv2.resize(frame,(width,height))
    frame = cv2.flip(frame,1)
    

    handData=findHands.Marks(frame)
    if train == 1:
        if handData !=[]:
            cv2.putText(frame,'Please Show Gesture ' + gestNames[trainCnt] + ': Press t when Ready',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
            #print('Please Show Gesture',gestNames[trainCnt],': Press t when Ready')
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
        cv2.putText(frame,'SHOW YOUR GESTURE',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        if handData !=[]:
            unknownGesture=findDistances(handData[0])
            myGesture=findGesture(unknownGesture,knownGestures,Keypoints,gestNames,tol)
            cv2.putText(frame,myGesture,(100,200),cv2.FONT_HERSHEY_SIMPLEX,3,(0,0,255),8)
            if myGesture:
                myGesture=myGesture+'\r'
                #arduinoData.write(myGesture.encode())
                
    
    for hand in handData:
        for ind in Keypoints:
            cv2.circle(frame,hand[ind],5,(255,0,0),3)

    cv2.imshow('MEXE 004 HGR', frame)
    cv2.moveWindow('MEXE 004 HGR',0,0)
    key = cv2.waitKey(1)
    if key == ord('q'):
            break  

cam.release()