#!/usr/bin/python3

#/usr/bin/python3 main.py

import tkinter as tk
from tkinter import Button, Canvas, Label, Frame
from PIL import ImageTk
import PIL.Image
#from src.gestureDetection import *
#from src.video import captureAndPredict

def main():
    root = tk.Tk()
    root.title("Welcome to LikeGeeks app")
    root.geometry('500x500')
    canvas = Canvas(root, height=500, width=500)
    canvas.pack()

    # background
    image = PIL.Image.open("images/bg.jpg")
    background_image = ImageTk.PhotoImage(image)
    background_label = Label(root, image=background_image)
    background_label.place(relwidth=1, relheight=1)



    label = tk.Label(root, text="Hello World!", font=("Arial Bold", 30)) # Create a text label
    label.place(relx=0.5, rely=0.15, relwidth=0.5, relheight=0.2, anchor='n')
    #label.pack(padx=20, pady=20) # Pack it into the window

    btn = Button(root, text="Search in a picture", bg="white", fg="black",command=clicked)
    btn.place(relx=0.5, rely=0.70, relwidth=1, relheight=0.05, anchor='center')

    btn2 = Button(root, text="Start recording", bg="white", fg="black", command=lambda: captureAndPredict)
    btn2.place(relx=0.5, rely=0.80, relwidth=1, relheight=0.05, anchor='center')


    root.mainloop()

def clicked():
    x = input('Write path:')
    try:
        return detectGesture(x)
    except Exception:
        return 

if __name__ == "__main__":
        main()
