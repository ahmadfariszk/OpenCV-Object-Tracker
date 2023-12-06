import cv2
import numpy as np 

cap = cv2.VideoCapture("los_angeles.mp4")

while True:
 _, frame = cap.read()    #cap.read() returns a tuple.

 cv2.imshow("Frame", frame)
 key = cv2.waitKey(0)      #returns the code of the key pressed
 if key == 27:             #escape key
  break

cap.release()