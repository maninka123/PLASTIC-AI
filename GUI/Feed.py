import cv2
import numpy as np
import time

#Select any model you want to use for inferencing
#net = cv2.dnn.readNet('YoloV4-Tiny/yoloV4_best.weights', 'YoloV4-Tiny/y4_tiny_cfg.cfg')
#net = cv2.dnn.readNet('YoloV4_Iterations_1000/yolov4-obj_last.weights', 'YoloV4_Iterations_1000/yolov4-obj.cfg')
net = cv2.dnn.readNet('YoloV3-Tiny/yolov3-tiny_best.weights', 'YoloV3-Tiny/yolov3-tiny.cfg')

classes = ["Plastic"]

#Bounding box Style
color = (204,204,12)
font = cv2.FONT_HERSHEY_PLAIN

#Predicting Function 
#Getting Inputs Camera image,REcording or not, Confidence level of the predictions
def detected_image(img,record_status,conf):
    height, width, _ = img.shape
    blob = cv2.dnn.blobFromImage(img, 1/255, (416, 416), (0,0,0), swapRB=True, crop=False)
    net.setInput(blob)
    output_layers_names = net.getUnconnectedOutLayersNames()
    layerOutputs = net.forward(output_layers_names)

    boxes = []
    confidences = []
    class_ids = []

    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.2:
                center_x = int(detection[0]*width)
                center_y = int(detection[1]*height)
                w = int(detection[2]*width)
                h = int(detection[3]*height)
                x = int(center_x - w/2)
                y = int(center_y - h/2)

                boxes.append([x, y, w, h])
                confidences.append((float(confidence)))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences,conf, 0.4)

    if len(indexes)>0:
        for i in indexes.flatten():
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            confidence = str(round(confidences[i],2))
            cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
            cv2.rectangle(img, (x-1,y-30), (x+200, y),color, -1)
            cv2.putText(img, label + " " + confidence, (x+1, y-5), font,2, (255,255,255),2)

    img=cv2.resize(img,(1003,503))
    if record_status==1:
        cv2.putText(img,"Rec", (15, 38), font,2, (0,0,255),2)
        cv2.line(img,(8,8),(8,38),(0,0,255),3)
        cv2.line(img,(8,8),(38,8),(0,0,255),3)
        
        cv2.line(img,(995,8),(995,38),(0,0,255),3)
        cv2.line(img,(995,8),(965,8),(0,0,255),3)
        
        cv2.line(img,(8,495),(8,465),(0,0,255),3)
        cv2.line(img,(8,495),(38,495),(0,0,255),3)
        
        cv2.line(img,(995,495),(995,465),(0,0,255),3)
        cv2.line(img,(995,495),(965,495),(0,0,255),3)

        
    return img,len(indexes)

