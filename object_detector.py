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

#blob blob (this is the part that detects the object)
blob = cv2.dnn.blobFromImage(img, 0.00392, (416,416), (0,0,0), True, crop=False)

net.setInput(blob)
outs = net.forward(output_layers)

cv2.imshow("Image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()