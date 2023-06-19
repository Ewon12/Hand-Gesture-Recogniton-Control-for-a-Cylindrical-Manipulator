import numpy as np
import cv2 as cv
from pathlib import Path


def get_image():
    Class = 'BACKWARD'
    Path('DATASET1/'+Class).mkdir(parents=True, exist_ok=True)
    width=1200
    height=720
    cam=cv.VideoCapture(0,cv.CAP_DSHOW)
    cam.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cam.set(cv.CAP_PROP_FRAME_HEIGHT,height)
    cam.set(cv.CAP_PROP_FPS, 30)
    cam.set(cv.CAP_PROP_FOURCC,cv.VideoWriter_fourcc(*'MJPG'))
    if not cam.isOpened():
        print("Cannot open camera")
        exit()
    i = 0
    while True:
        ret, frame = cam.read()
        frame = cv.resize(frame,(width,height))

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break
        frame = cv.flip(frame,1)
        i+= 1
        if i % 5==0:
            cv.imwrite('DATASET1/'+Class+'/'+str(i)+'.png',frame)
   
        cv.imshow('frame', frame)
        if cv.waitKey(1) == ord('q') or i > 500:
            break
  
    cam.release()
    cv.destroyAllWindows()
if __name__ == "__main__":
   get_image()
  