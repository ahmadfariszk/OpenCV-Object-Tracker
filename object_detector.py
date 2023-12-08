#this file is writtento better understand how object detection in this project works

import cv2
import numpy as np

#Load YOLO
net = cv2.dnn.readNet("source_code\dnn_model\yolov4.weights", "source_code\dnn_model\yolov4.cfg")
classes = []
with open("source_code\dnn_model\classes.txt", "r") as f:
    classes = [line.strip() for line in f.readlines()]   #put file contains into an array, by line
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]

#Load image
img = cv2.imread("cycling001-1024x683.jpg")
height, width, channels = img.shape

#blob blob (this is the part that detects the object)
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)
net.setInput(blob)
outs = net.forward(output_layers)

#Framing the detected objects with identifiers
boxes= []           #lists declarations where we will store all the data for eeach detected object
confidences = []
class_ids = []
for out in outs:               #iterate through the outputs of all layers (outs)
    for detection in out:      #iterate through all the detected objects within each layer
        scores = detection[5:]
        class_id = np.argmax(scores)   #to find what class is the detected object
        confidence = scores[class_id]  #calculate confidence
        if confidence > 0.5:           #threshold for when object is count as "detected"
            center_x = int(detection[0] * width)
            center_y = int(detection[1] * height)
            w = int(detection[2] * width)
            h = int(detection[2] * height)
            
            #checking what object are detected and where the centre opf the object is. uncomment to make it work
            # cv2.circle(img,(center_x, center_y), 10, (0,255,0),2)
            
            x = int (center_x - w/2) #object rectangle
            y = int (center_y - h /2)
            boxes.append([x,y,w,h])                 #adds object data to each list
            confidences.append(float(confidence))
            class_ids.append(class_id)

detected_objects_num = len (boxes)
for i in range(len(boxes)): #create boxes
    x,y,w,h=boxes[i]
    label = classes[class_ids[i]]
    cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0))
    cv2.putText(img, label, (x,y+30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 1)
cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()