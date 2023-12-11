import math
import cv2
import numpy as np 
from object_detection import ObjectDetection

#Initialise object detection
od = ObjectDetection()

cap = cv2.VideoCapture("los_angeles.mp4")
frame_counter = 0
centerpoints_previous = [] #first frame have no previous poinr, so its blank
objects_tracked = {}
track_counter = 0    #counter used to assign new  entries to \object_tracked dictionary

while True:
  _, frame = cap.read()    #cap.read() returns a tuple.
  frame_counter += 1
  if not _:
    break

  centerpoints_current = []

  #object detection
  (class_ids, scores, boxes) = od.detect(frame)
  for box in boxes: #creating boxes for the objects
    (x,y,w,h) = box
    center_x = int((x+x + w)/2)    #find centre of box
    center_y = int((y+y + h)/2)
    centerpoints_current.append((center_x, center_y))
    #print("FRAME NO", frame_counter, "",x,y,w,h)   #debug
    cv2.rectangle(frame, (x,y), (x+w,y+h), (0, 255, 0), 2)

  #begin tracking between frames
  if frame_counter <=1:
    for point1 in centerpoints_current:            #point 1 is from the current frame
      for point2 in centerpoints_previous:         #point 2 id from the previous frame
        distance = math.hypot(point2[0]-point1[0], point2[1]- point1[1]) #determine distance with previous frame point
        #print("point1 ", point1[0],  point1[1], "point2 ", point2[0], point2[1],"distance ", distance) #debug
        if distance < 30:    #puting the previous and current points as the same object between frames when it is close together
          objects_tracked[track_counter] = point1
          track_counter += 1
  else:
    objects_tracked_copy = objects_tracked.copy() 
    #the dictionary is copied so that items can be removed (from the original) while looping through the copy
    for point1 in centerpoints_current:
      object_exists = False
      for object_id, point1 in objects_tracked_copy.items():  #object id is the persistent (through diff frames) id ob tracked object
        distance = math.hypot(point2[0]-point1[0], point2[1]- point1[1])
        if distance < 30: #check if the same object
          objects_tracked[object_id] = point1 #update object position
          object_exists = True
          continue
      if not object_exists:
        objects_tracked.pop(object_id)


  for object_id, point1 in objects_tracked.items():
    cv2.circle(frame, point1, 5, (0,0,255), -1)  #draw points
    cv2.putText(frame, str(object_id), (point1[0], point1[1]-7), 0, 1, (0,0,255), 1)

    
  #debug
  print("Tracked objects ")
  print(objects_tracked)
  print("Current Frame Points ")
  print(centerpoints_current)
  print("Previous Frame Points ")
  print(centerpoints_previous)

  centerpoints_previous = centerpoints_current.copy() 

  cv2.imshow("Frame", frame)
  key = cv2.waitKey(0)      #returns the code of the key pressed
  if key == 27:             #escape key
    break

cap.release()
cv2.destroyAllWindows()