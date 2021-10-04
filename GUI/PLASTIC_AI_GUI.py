#PROJECT PLASTIC AI
#G17 University Ambassadors Consortium
#University of Moratuwa
#SDG Goal 14 - Life below water
#P M Ranasinghe
#11/10/2021
#%%

#GUI for the PLASTIC AI Project

#Importing the required libraries
import cv2
import datetime 
from datetime import datetime
import Feed
import tkinter as tk
import imutils
from PIL import Image,ImageTk
from tkinter import ttk
from tkinter import filedialog
import time

#Starting the Video
cap = cv2.VideoCapture(0)

#Classes to be predicted
classes = ["Plastic"]

#Bounding box design
color = (204,204,12)
font = cv2.FONT_HERSHEY_PLAIN

pTime=0

#Recording the Live feed
record_status=0
now=datetime.now()
dt = now.strftime("%d_%m_%Y__%H_%M_%S")
name="Savings/"+dt+" Feed"+".avi"
Video_save = cv2.VideoWriter(name,cv2.VideoWriter_fourcc(*'MJPG'), 10, (1003,503))

#Confidence level of the model
conf=50

#%%Creating the Interface
background_colour="#D7E9F7"

#Basic Window
win=tk.Tk()
win.title("PLASTIC AI")
win.iconbitmap("favicon.ico")
win.minsize(1075,750)
win.config(padx=20,pady=20,bg=background_colour)

#Required Functions
#Function to resize images to fit into the Interface 
def img_resize(img):
    h,w=img.shape[0:2]
    if (h>w):
        img=imutils.resize(img,height=400)
    else:
        img=imutils.resize(img,width=1000)
    return img    

#Function to upload videos to the GUI
def UploadAction():
    global image_uploaded,filename
    try:
        filename = filedialog.askopenfilename(title = "Select a File",filetypes=[('Video Files', '*mp4'),('Video Files', '*avi'),('Video Files', '*mkv'),('Video Files', '*mov'),('Video Files', '*wmv'),('Video Files', '*m4v')])
        if(len(filename)) >2:
            return filename
        else:
            print("Invalid Selection")
    except:
        pass
    
#Loop to play the Video in the Interface    
def update():
        global u,pTime,record_status,conf
        ret, frame = cap.read()
        if (ret==True):
            img_box,count=Feed.detected_image(frame,record_status,conf/100)
            img_coloured=cv2.cvtColor(img_box,cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img_coloured)
            imgtk = ImageTk.PhotoImage(image=img)
            label_camera.imgtk = imgtk
            label_camera.configure(image=imgtk)
            fTime=time.time()
            fps=1/(fTime-pTime)
            pTime=fTime
            fps_show["text"]=f" {round(fps,1)} "
            Count_show["text"]=f" {count} "
            if (record_status==1):
                Video_save.write(img_box)
            u=canvas_1.after(1, update)

#Function to Initialize the Camera/display in the GUI        
def Strat_camera():
    global label_camera
    start_button["state"]=tk.DISABLED
    stop_button["state"]=tk.NORMAL
    status_show["fg"]="green"
    label_camera.place(x=0, y=0)
    update()
    
#Function to Stop the Feed/Video  
def Stop_feed():
    canvas_1.after_cancel(u)
    start_button["state"]=tk.NORMAL
    stop_button["state"]=tk.DISABLED
    status_show["fg"]="red"
    fps_show["text"]=" 0 "
    Count_show["text"]=" 0 "
    
#Function to Change the REcording Statues to 0 or 1
def record():
    global record_status
    if (record_status==0):
        record_status=1
    else:
        record_status=0

#Function to Select the iinput mode from the drop down list    
def option_changed(event):
    global cap
    if (r_down.get()=="Inbuilt Camera"):
        cap = cv2.VideoCapture(0)
    if (r_down.get()=="Custom Video"):
        file=UploadAction()
        cap = cv2.VideoCapture(file)

#Function to Putup the First image to the GUI
def img_open(path):
    if (type(path)==str):
        print('Selected:', path)
        test_img=cv2.imread(path)
    else:
        test_img=path
    resized_img=img_resize(test_img)
    resized_img = cv2.cvtColor(resized_img, cv2.COLOR_BGR2RGB)
    im_pil = Image.fromarray(resized_img)
    test_img=ImageTk.PhotoImage(im_pil)
    return test_img

#Function to Change the Slider values of the Confidace level         
def slide_conf(q):
    global conf
    conf=int(q)

#Name the Window
gui_label=tk.Label(text="PLASTIC AI",font=("Arial",26,"bold"),bg=background_colour)
gui_label.place(x=10,y=0)

#Creating the Video feed canvas
canvas_1=tk.Canvas(win,width=1000,height=500,bg=background_colour,highlightthickness=5)
canvas_1.place(x=12,y=127)

#Adding a label to show the camera
label_camera = tk.Label(canvas_1)

#Loading the initiale image into the GUI
default_img_path="ABC.jpeg"
default_img=img_open(default_img_path)
imported_img=canvas_1.create_image(505,250, image=default_img)

#Select the Input method label
camera_label=tk.Label(win,text="Select Camera",bg=background_colour,font=("Arial",14,"bold"))
camera_label.place(x=10,y=80)

#Select the Input method drop down menu
r_down=tk.StringVar()
drop_down_items=["Inbuilt Camera","External Camera","Custom Video"]
drop_down=tk.OptionMenu(win,r_down,*drop_down_items,command=option_changed)
r_down.set("Inbuilt Camera")
drop_down.place(x=170,y=80)
#drop_down['menu'].entryconfigure(1,state =tk.DISABLED)

#Start the feed
start_button=tk.Button(win,text="Start",font=("Arial",12,"bold"),padx=4,pady=2,width=15,command=Strat_camera,state=tk.NORMAL)
start_button.place(x=140,y=650)

#Stop the feed
stop_button=tk.Button(win,text="Stop",font=("Arial",12,"bold"),padx=4,pady=2,width=15,command=Stop_feed,state=tk.DISABLED)
stop_button.place(x=340,y=650)

#Strat/Stop Recording
record_button=tk.Button(win,text="Record",bg="red",fg="black",font=("Arial",12,"bold"),padx=4,pady=2,width=15,command=record,state=tk.NORMAL)
record_button.place(x=540,y=650)

#Label of University Consortium
group_label=tk.Label(text="© G17 University Ambassadors Consortium",font=("Arial",10,"bold"),bg=background_colour)
group_label.place(x=775,y=700)

#Status of the GUI Predicting or Ideal
status_label=tk.Label(win,text="STATUS",bg=background_colour,font=("Arial",12))
status_label.place(x=900,y=5)

status_show=tk.Label(win,text=" 🔴 ",fg="red",bg=background_colour,font=("Arial",12,"bold"))
status_show.place(x=970,y=3)

#Displaying FPS Value
fps_label=tk.Label(win,text="FPS",bg=background_colour,font=("Arial",12))
fps_label.place(x=900,y=50)

fps_show=tk.Label(win,text=" 0 ",font=("Arial",12,"bold"))
fps_show.place(x=970,y=50)

#Counting the plastic in the given frame
Count_label=tk.Label(win,text="COUNT",bg=background_colour,font=("Arial",12))
Count_label.place(x=900,y=80)

Count_show=tk.Label(win,text=" 0 ",font=("Arial",12,"bold"))
Count_show.place(x=970,y=80)

#Change the confidence of the model accordingly
slider=tk.Scale(label='Confidence',from_=2,to=99,bg=background_colour,tickinterval=0.1,orient="horizontal",command=slide_conf)
slider.place(x=760,y=5)
slider.set(conf)


win.mainloop()
cap.release()


    