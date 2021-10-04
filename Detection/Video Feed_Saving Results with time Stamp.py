#PROJECT PLASTIC AI
#G17 University Ambassadors Consortium
#University of Moratuwa
#SDG Goal 14 - Life below water
#P M Ranasinghe
#11/10/2021

#Code to detect platic from the camera or a live-video feed + Time Stamp + Location data 

#Importing the required libraries
import cv2
import numpy as np
import time
from datetime import datetime
import json
import piexif
import piexif.helper
import geocoder

#Select any model you want to use for inferencing
#net = cv2.dnn.readNet('YoloV4-Tiny/yoloV4_best.weights', 'YoloV4-Tiny/y4_tiny_cfg.cfg')
#net = cv2.dnn.readNet('YoloV4_Iterations_1000/yolov4-obj_last.weights', 'YoloV4_Iterations_1000/yolov4-obj.cfg')
net = cv2.dnn.readNet('YoloV3-Tiny/yolov3-tiny_best.weights', 'YoloV3-Tiny/yolov3-tiny.cfg')

#Classes to be predicted
classes = ["Plastic"]

#Font and Color to be use in drawing bounding boxes
color=(204,204,12)
font = cv2.FONT_HERSHEY_PLAIN

#A function to get live location -Longitude and latitude (Internet needed)
def get_live_location():
    live_loc = geocoder.ip('me')
    Lat,Long=live_loc.latlng
    return Lat,Long

#A function to change edit image meta_data accordingly
#Add information according to your application
def Change_meta_data(filename,dt,Lat,Long,CAMERA_NO="000 001"):
    userdata={'Category':'Plastic detected',
          'Taken by Camera No':CAMERA_NO,
          'Date':dt,
          'Location':f"{(Lat,Long)}" }

    exif_dict = piexif.load(filename)
    exif_dict["Exif"][piexif.ExifIFD.UserComment] = piexif.helper.UserComment.dump(
    json.dumps(userdata),encoding="unicode")
    
    piexif.insert(piexif.dump(exif_dict),filename)

#A function to add time stamp to the given image
#Longitude and Latitude canbe pass as a input as a sensor signal
#or get the live location with geocorder library
def Capture_evidence(img,lat=6.927079,long=79.861244):
    #lat,long=get_live_location() #Comment this if you are using using a gps sensor to get location data
    col=(0,0,255) #Stamp Colour (Red)
    H=img.shape[0]
    now=datetime.now()
    dt = now.strftime("%Y-%m-%d %H %M %S")
    cv2.rectangle(img,(5,H-85),(320, H-4),(0,0,255), 2)
    cv2.putText(img,str(dt),(10,H-60),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
    cv2.putText(img,f"long: {long}",(10,H-35),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
    cv2.putText(img,f"lat  : {lat}",(10,H-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,col,2)
    file_name="Saved/"+dt+".jpg"
    cv2.imwrite(file_name, img)
    Change_meta_data(file_name,dt,long,lat)
    return img

#Start Camera or live video feed 
# 0 - Webcam (Inbuilt camera)
# 1 - External Camera
# You can specify the path to video feed as well    
cap = cv2.VideoCapture("Y2Mate.is - Stunning New Video Shows Massive Plastic Debris In Ocean   NBC Nightly News online video cutter com-QMe00iN5y_E-720p-1630566090031.mp4")
pTime=0

#Starting Prediction loop
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
                #Adding Timestamp + Changing image meta data
                if (label=="Plastic"):
                    img=Capture_evidence(img)

        cTime=time.time()
        fps=1/(cTime-pTime)
        pTime=cTime
        cv2.putText(img,str(int(fps))+" fps",(10,35),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,0),2)
    
        cv2.imshow('Video Feed', img)
        key = cv2.waitKey(1)
        #Press ESC to exit from the video
        if key==27:
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()