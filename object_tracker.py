import cv2
import numpy as np 
from object_detection import ObjectDetection

#Initialise object detection
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")

while True:
  _, frame = cap.read()    #cap.read() returns a tuple.
  if not _:
    break
  #object detection
  (class_ids, scores, boxes) = od.detect(frame)
  for box in boxes: #creating boxes for the objects
    (x,y,w,h) = box
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)

  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1)      #returns the code of the key pressed
  if key == 27:             #escape key
    break

cap.release()
cv2.destroyAllWindows()