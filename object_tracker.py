import cv2
import numpy as np 
from object_detection import ObjectDetection

#Initialise object detection
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")
frame_counter = 0
center_points = [ ]

while True:
  _, frame = cap.read()    #cap.read() returns a tuple.
  frame_counter += 1
  if not _:
    break

  #object detection
  (class_ids, scores, boxes) = od.detect(frame)
  for box in boxes: #creating boxes for the objects
    (x,y,w,h) = box
    center_x = int((x + x + w)/2)    #find centre of box
    center_y = int((y + y + h)/2)
    center_points.append((center_x, center_y))
    print("FRAME NO", frame_counter, "",x,y,w,h)
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)

  for point in center_points:
    cv2.circle(frame, point, 5, (0,0,255), -1)

  cv2.imshow("Frame", frame)
  key = cv2.waitKey(1)      #returns the code of the key pressed
  if key == 27:             #escape key
    break

cap.release()
cv2.destroyAllWindows()