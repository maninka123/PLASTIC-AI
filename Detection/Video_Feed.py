#PROJECT PLASTIC AI
#G17 University Ambassadors Consortium
#University of Moratuwa
#SDG Goal 14 - Life below water
#P M Ranasinghe
#11/10/2021

#Code to detect platic from the camera or a live-video feed

#Importing the required libraries
import cv2
import numpy as np
import time

#Select any model you want to use for inferencing
#net = cv2.dnn.readNet('YoloV4-Tiny/yoloV4_best.weights', 'YoloV4-Tiny/y4_tiny_cfg.cfg')
#net = cv2.dnn.readNet('YoloV4_Iterations_1000/yolov4-obj_last.weights', 'YoloV4_Iterations_1000/yolov4-obj.cfg')
net = cv2.dnn.readNet('YoloV3-Tiny/yolov3-tiny_best.weights', 'YoloV3-Tiny/yolov3-tiny.cfg')

#Classes to be predicted
classes = ["Plastic"]

#Font and Color to be use in drawing bounding boxes
font = cv2.FONT_HERSHEY_PLAIN
color = (204,204,12)

#Start Camera or live video feed 
# 0 - Webcam (Inbuilt camera)
# 1 - External Camera
# You can specify the path to video feed as well
cap = cv2.VideoCapture("Y2Mate.is - Stunning New Video Shows Massive Plastic Debris In Ocean   NBC Nightly News online video cutter com-QMe00iN5y_E-720p-1630566090031.mp4")
pTime=0

#Staring prediction loop
while True:
    ret, img = cap.read()
    if ret:
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
    
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    
        if len(indexes)>0:
            for i in indexes.flatten():
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                confidence = str(round(confidences[i],2))
                cv2.rectangle(img, (x,y), (x+w, y+h), color, 2)
                cv2.rectangle(img, (x-1,y-30), (x+200, y),color, -1)
                cv2.putText(img, label + " " + confidence, (x+1, y-5), font,2, (255,255,255),2)
        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps))+" fps",(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    
        cv2.imshow('Video Feed', img)
        key = cv2.waitKey(1)
        
        #Press ESC to close the video
        if key==27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()