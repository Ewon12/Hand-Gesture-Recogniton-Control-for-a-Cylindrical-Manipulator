import cv2
import numpy as np
import pickle
import time
import serial
import PySimpleGUI as sg

sg.theme('DarkBrown4')

intro = [
   
    [sg.Push(),sg.Image(filename='hgrc.png',size = (720,530)),sg.Push()], [sg.VerticalSeparator(pad=(0, 10))],
    [sg.Push(),sg.Button('RUN', disabled=False ,font=("LCDMono2 Bold",30),button_color=('black','pink')),sg.Push()],
    [sg.Push(),sg.Exit(font = ('Times New Roman',13))]

]




def main_layout():
    
    sg.theme('DarkBrown4')

    Main_layout = [
        [sg.Push(),sg.Button('HELP', disabled=False ,font=("Arial Bold",10),button_color=('black','white')),
        sg.Button('ABOUT', disabled=False ,font=("Arial Bold",10),button_color=('black','white'))],
        [sg.Push(),sg.Text('HAND GESTURE RECOGNITION CONTROL FOR A CYLINDRICAL MANIPULATOR',
        font =("Dungeon Bold",20),text_color='orange'),sg.Push()],[sg.VerticalSeparator(pad=(0, 10))],
        
            [sg.Push(),sg.Frame('Fill out the following: ',[
                [sg.Text('Filename for training data: ', font = ('Times New Roman',15)),sg.InputText('default',key='default',size=(15,10))],
                [sg.Text('SERIAL PORT: ', font = ('Times New Roman',15)),sg.InputText('COM5',key='SERIAL',size=(15,10))],
                [sg.Text('CAMERA: ', font = ('Times New Roman',15)),sg.InputText('0',key='CAM',size=(15,10))],
                [sg.Text('HEIGHT for FRAME: ', font = ('Times New Roman',15)),sg.InputText('640',key='height',size=(15,10))],
                [sg.Text('WIDTH for FRAME: ', font = ('Times New Roman',15)),sg.InputText('920',key='width',size=(15,10))]]),sg.Push(),
            sg.Frame('GESTURE AVAILABLE:',[
                [sg.Text('UP ', font = ('Times New Roman',15)),
                sg.Text('DOWN ', font = ('Times New Roman',15))],
                [sg.Text('RIGHT ', font = ('Times New Roman',15)),
                sg.Text('LEFT ', font = ('Times New Roman',15))],
                [sg.Text('FORWARD ', font = ('Times New Roman',15)),
                sg.Text('BACKWARD ', font = ('Times New Roman',15))],
                [sg.Text('GRIP ', font = ('Times New Roman',15)),
                sg.Text('UNGRIP ', font = ('Times New Roman',15))],
                [sg.Text('RESET ', font = ('Times New Roman',15)),
                sg.Text('GO AUTO ', font = ('Times New Roman',15))]]),sg.Push()],
            [sg.VerticalSeparator(pad=(0, 10))],
            [sg.Push(),sg.Button('RECOGNIZE', disabled=False ,font=("LCDMono2 Bold",20),button_color=('orange','purple')),sg.Push(),
            sg.Button('UPLOAD', disabled=False ,font=("LCDMono2 Bold",20),button_color=('blue','green')),sg.Push(),
            sg.Button('TRAIN', disabled=False ,font=("LCDMono2 Bold",20),button_color=('yellow','brown')),sg.Push()],
            [sg.VerticalSeparator(pad=(0, 10))],
            [sg.Push(),sg.Exit(font = ('Times New Roman',13))]

    ]

    def Help_windows():
        sg.theme('DarkBrown4')
        Helps_layout =[
            [sg.Push(),sg.Text('INSTRUCTION',font =("Cooper Black",20)),sg.Push()],
            [sg.Text('a.) Before pressing the Recognize or Train or Upload button, you must answer the requirements in \"fill out the following\".',font =("Arial Bold",15)),sg.Push()],
            [sg.Text('For the \"FIlename for trainning data\", type of desired filename if Train, while for Recognize type the saved filename.',font =("Arial Bold",10)),sg.Push()],
            [sg.Text('For the \"SERIAL PORT\", type the location \"COM PORT\" of arduino',font =("Arial Bold",10)),sg.Push()],
            [sg.Text('For the \"CAMERA\", type the number of desired camera used(which is 0,1..... so on).',font =("Arial Bold",10)),sg.Push()],
            [sg.Text('For the \"HEIGHT for FRAME\", type desired pixel for height',font =("Arial Bold",10)),sg.Push()],
            [sg.Text('For the \"WIDTH for FRAME\", type desired pixel for width',font =("Arial Bold",10)),sg.Push()],
            [sg.Text('b.)If clicked TRAIN, the camera will be opened, pressed \"t\" if the hand posture is ready. ',font =("Arial Bold",15)),sg.Push()],
            [sg.Text('Popup \"DONE\" if finish and you can close the camera. Additionally, pressed \"q\" if stop the train.',font =("Arial Bold",15)),sg.Push()],
            [sg.Text('c.)If clicked Recognize, the camera will be opened, when show hand posture based on train it will shows the name as available gesture. ',font =("Arial Bold",15)),sg.Push()],
            [sg.Text('Then uploading to arduino to every shows hand posture. Additionally, pressed \"q\" if stop the recognize',font =("Arial Bold",15)),sg.Push()],
            [sg.Text('d.) The command check is uploading a type words from available gesture to testing the arduino by click the \"UPLOAD\". ',font =("Arial Bold",15)),sg.Push()],
            [sg.Push(),sg.Exit(font = ('Times New Roman',13))]
            
        ]
        Help_windows =sg.Window('HELP',Helps_layout)
        while True:
            event, values = Help_windows.read()
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
        Help_windows.close()
        
    def About_windows():
        
        sg.theme('DarkBrown4')
        about_layout=[
            [sg.Push(),sg.Text('CREATED BY MECHATRONICS ENGINEERING STUDENTS:',font =("Cooper Black",20)),sg.Push()],
            [sg.Push(),sg.Text('MEXE - 004',font =("Time New Roman",15)),sg.Push()],
            [sg.Push(),sg.Exit(font = ('Times New Roman',13))]
        ]
        
        about_windows =sg.Window('ABOUT',about_layout)
        while True:
            event, values = about_windows.read()
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
        about_windows.close()
        
    def upload_windows():
        sg.theme('DarkBrown4')
            
        upload_layout=[
            [sg.Push(),sg.Frame('COMMAND CHECK:',[
            [sg.Text('TESTING COMMAND: ', font = ('Times New Roman',15)),sg.InputText('',key='upload',size=(15,10))]]),sg.Push()],
            [sg.Push(),sg.Button('UPLOAD', disabled=False ,font=("Rockwell Extra Bold",15),button_color=('blue','green')),sg.Push()],
            [sg.Push(),sg.Exit(font = ('Times New Roman',13))]
            ]
            
        upload_windows =sg.Window('UPLOAD', upload_layout)
        while True:
            event, values = upload_windows.read()
            if event == sg.WIN_CLOSED or event == 'Exit':
                arduinoData.close()
                arduinoData.open()  # Reopen the connection
                arduinoData.setDTR(False)  # Reset Arduino
                arduinoData.close()

                break
            elif event == "UPLOAD":
                    Command=values['upload']
                    Commands = Command + '\r'
                    arduinoData.write(Commands.encode())




        upload_windows.close()
        arduinoData.close()
        arduinoData.open()  # Reopen the connection
        arduinoData.setDTR(False)  # Reset Arduino
        arduinoData.close()


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

    def findGesture(unknownGesture, knownGestures, Keypoints, gestNames, tol, return_distance=False):
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
        if return_distance:
            return gesture, errorMin
        else:
            return gesture
    tol = 50
    cTime=0
    pTime=0
    window = sg.Window('HAND GESTURE RECOGNITION CONTROL', Main_layout,resizable =False)

    while True:
            event, values =window.read()
            if event == sg.WIN_CLOSED or event == 'Exit':
                break
            elif event == 'TRAIN':
                window.hide()
                width=int(values["width"])
                height=int(values["height"])
                cam=cv2.VideoCapture(int(values["CAM"]),cv2.CAP_DSHOW)
                
                if not cam.isOpened():
                    sg.popup(f"Camera {values['CAM']} is not available.")
                    continue
                    

                else:
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cam.set(cv2.CAP_PROP_FPS, 60)
                    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                    
                findHands = mpHands(1)
                time.sleep(3)
                Keypoints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                trainCnt=0
                knownGestures=[]
                numGest= 10
                gestNames=['UP', 'DOWN', 'LEFT', 'RIGHT', 'FORWARD', 'BACKWARD', 'GRIP', 'UNGRIP', 'RESET','Go Auto']
                trainName=values['default']
                if trainName=='':
                    trainName='default'
                trainName=trainName+'.pkl'
                    
                while True:
                    ignore, frame = cam.read()
                    frame = cv2.resize(frame,(width,height))
                    frame = cv2.flip(frame,1)
                    
                    cTime = time.time()
                    fps=1/(cTime-pTime)
                    pTime=cTime
                    cv2.putText(frame, str(int(fps)), (800,70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),3)
                                

                    handData=findHands.Marks(frame)

                    if handData !=[]:
                        cv2.putText(frame,'Please Show Gesture ' + gestNames[trainCnt] + ': Press t when Ready',(50,100),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
                        #print('Please Show Gesture',gestNames[trainCnt],': Press t when Ready')
                        if cv2.waitKey(1)  & 0xff == ord('t'):
                            knownGesture=findDistances(handData[0])
                            knownGestures.append(knownGesture)
                            trainCnt=trainCnt+1
                            if trainCnt==numGest:
                                train=0
                                with open(trainName, 'wb') as f:
                                    pickle.dump(gestNames,f)
                                    pickle.dump(knownGestures,f)
                                sg.popup('DONE')
                                cam.release()
                                cv2.destroyAllWindows()
                                break
                                
                                    
                    for hand in handData:
                        for ind in Keypoints:
                            cv2.circle(frame,hand[ind],5,(255,0,0),3)

                    cv2.imshow('MEXE 004 HGR', frame)
                    cv2.moveWindow('MEXE 004 HGR',0,0)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                            break  

                cam.release()
                cv2.destroyAllWindows()
                window.un_hide()
            
            elif event == 'RECOGNIZE':
                window.hide()
                try:
                    arduinoData=serial.Serial((values['SERIAL']),9600)
                except serial.SerialException:
                    sg.popup("Error: could not connect to Arduino board")
                    window.un_hide()
                    continue
                    
                width=int(values["width"])
                height=int(values["height"])
                cam=cv2.VideoCapture(int(values["CAM"]),cv2.CAP_DSHOW)
                
                if not cam.isOpened():
                    sg.popup(f"Camera {values['CAM']} is not available.")
                    window.un_hide()
                    continue
                    
                else:
                    cam.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                    cam.set(cv2.CAP_PROP_FPS, 60)
                    cam.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
                findHands = mpHands(1)
                time.sleep(3)
                Keypoints = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
                trainName=values['default']
                if trainName=='':
                    trainName='default'
                trainName=trainName+'.pkl'
                
                try:
                    with open(trainName,'rb') as f:
                        gestNames=pickle.load(f)
                        knownGestures=pickle.load(f)
                except FileNotFoundError:
                    cam.release()
                    sg.popup(f"The file '{trainName}' is not available. Make sure filename is correct spell.")
                    
                    continue
                
                # Initialize variables
                start_time = time.time()
                last_display_time = start_time
                last_send_time = start_time
                time_interval = 5

                while True:
                    ignore, frame = cam.read()
                    frame = cv2.resize(frame,(width,height))
                    frame = cv2.flip(frame,1)
                    
                    cTime = time.time()
                    fps=1/(cTime-pTime)
                    pTime=cTime
                    cv2.putText(frame, str(int(fps)), (abs(width-800),70),cv2.FONT_HERSHEY_PLAIN,3,(255,255,0),3)
                                
                    handData=findHands.Marks(frame)
                    cv2.putText(frame, 'SHOW YOUR GESTURE', (25, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                        
                    if handData:
                        unknownGesture = findDistances(handData[0])
                        myGesture, dist = findGesture(unknownGesture, knownGestures, Keypoints, gestNames, tol, return_distance=True)
                        
                        # Calculate accuracy and precision
                        accuracy = round((tol - dist) / tol * 100, 2) if dist <= tol else 0.0
                        precision = round((tol - dist) / tol * len(myGesture)*100, 2) if dist <= tol else 0.0
                        recall = round(len(myGesture) / len(knownGestures), 2)
                        f1 = round(2 * (precision * recall) / (precision + recall)*100, 2) if (precision + recall) > 0 else 0.0
                        # Display the recognized gesture with accuracy, precision and f1 score
                        cv2.putText(frame, f"{myGesture} ( ACCURACY: {accuracy}%, PRECISION: {precision}%)", (25, 125), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 0, 0), 3)
                        
                        if myGesture in ['UP','DOWN','LEFT','RIGHT','FORWARD','BACKWARD','GRIP','UNGRIP','RESET','Go Auto']:
                            # Introduce a delay between sending each message
                            current_time = time.time()
                            if (current_time - last_display_time) >= time_interval:
                                # Display the time in seconds on the screen
                                seconds_elapsed = int(current_time - start_time)
                                time_remaining = time_interval - seconds_elapsed
                                text = f"Time: {time_remaining} sec"
                                cv2.putText(frame, text, (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                                
                                # Reset the timer
                                start_time = current_time
                                last_display_time = current_time
                                
                            else:
                                # If the time interval has not elapsed, display the previous time remaining
                                seconds_elapsed = int(current_time - start_time)
                                time_remaining = time_interval - seconds_elapsed
                                text = f"Time: {time_remaining} sec"
                                cv2.putText(frame, text, (25, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2)
                                


                            if (current_time - last_send_time) >= time_interval:
                                # Send the gesture to Arduino
                                myGesture += '\r'
                                arduinoData.write(myGesture.encode())
                                last_send_time = current_time

                    for hand in handData:
                        for ind in Keypoints:
                            cv2.circle(frame,hand[ind],5,(255,0,0),3)

                    cv2.imshow('MEXE 004 HGR', frame)
                    cv2.moveWindow('MEXE 004 HGR',0,0)
                    key = cv2.waitKey(1)
                    if key == ord('q'):
                            break
                cam.release()
                cv2.destroyAllWindows()
                arduinoData.close()
                arduinoData.open()  # Reopen the connection
                arduinoData.setDTR(False)  # Reset Arduino
                arduinoData.close()
                window.un_hide()
        
            elif event == 'UPLOAD':
                window.hide()
                try:
                    arduinoData=serial.Serial((values['SERIAL']),9600)
                except serial.SerialException:
                    sg.popup("Error: could not connect to Arduino board")
                    window.un_hide()
                    continue
                upload_windows()
                window.un_hide()

            
            elif event == 'HELP':
                window.hide()
                Help_windows()
                window.un_hide()
            
            elif event == 'ABOUT':
                window.hide()
                About_windows()
                window.un_hide()
    window.close()

intro_windows =sg.Window('MEXE 004',intro)
while True:
        event, values =intro_windows.read()
        if event == sg.WIN_CLOSED or event == 'Exit':
            break
        
        elif event == 'RUN':
            intro_windows.hide()
            main_layout()
            intro_windows.un_hide()

intro_windows.close()        