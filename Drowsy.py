import numpy as np
import cv2  
import winsound

eye_detect=cv2.CascadeClassifier('haarcascade_eye.xml')
eye_detect.load('haarcascade_eye.xml')
frequency = 2500
duration = 1000
cap=cv2.VideoCapture(0)
while (True):
    state,pix=cap.read()
    if not state:
        print('No frame captured')
        
    pix=cv2.resize(pix,(640,480))    
    eyes=eye_detect.detectMultiScale(pix,scaleFactor=1.1,minSize=(50,50),minNeighbors=5,flags=cv2.CASCADE_SCALE_IMAGE)
    if(len(eyes)==0):
        winsound.Beep(frequency, duration)
    for (x,y,w,h) in eyes:
        cv2.rectangle(pix,(x,y),(x+w,y+h),(0,255,0),2)

    # cv2.imshow('img',pix)
    cv2.imshow('Detection',pix)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()