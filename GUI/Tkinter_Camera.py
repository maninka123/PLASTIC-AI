import PIL
from PIL import Image,ImageTk
import pytesseract
import cv2
from tkinter import *
width, height = 800, 600
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)

root = Tk()
root.bind('<Escape>', lambda e: root.quit())
lmain = Label(root)
lmain.pack()

def show_frame():
    _, frame = cap.read()
    frame = cv2.flip(frame, 1)
    cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    img = PIL.Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    lmain.imgtk = imgtk
    lmain.configure(image=imgtk)
    lmain.after(10, show_frame)

show_frame()
root.mainloop()
#%%
import tkinter as tk
from PIL import Image, ImageTk
import cv2

class MainWindow():
    def __init__(self, window, cap):
        self.window = window
        self.cap = cap
        self.width = self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        self.interval = 20 # Interval in ms to get the latest frame

        # Create canvas for image
        self.canvas = tk.Canvas(self.window, width=self.width, height=self.height)
        self.canvas.grid(row=0, column=0)

        # Update image on canvas
        self.update_image()

    def update_image(self):
        # Get the latest frame and convert image format
        self.image = cv2.cvtColor(self.cap.read()[1], cv2.COLOR_BGR2RGB) # to RGB
        self.image = Image.fromarray(self.image) # to PIL format
        self.image = ImageTk.PhotoImage(self.image) # to ImageTk format

        # Update image
        self.canvas.create_image(0, 0, anchor=tk.NW, image=self.image)

        # Repeat every 'interval' ms
        self.window.after(self.interval, self.update_image)

if __name__ == "__main__":
    root = tk.Tk()
    MainWindow(root, cv2.VideoCapture(0))
    root.mainloop()