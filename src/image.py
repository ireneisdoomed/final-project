
from keras.preprocessing.image import ImageDataGenerator 
import cv2
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from resizeimage import resizeimage


def readFrame(sec, path, count):
    '''
    Reads a video and saves each of its frames in .jpg format
    input: timestamp of the video, path of the video, number of the frame
    output: instance of readFrame
    '''
    # Current position of the video file in milliseconds.
    vidcap = cv2.VideoCapture(path)
    vidcap.set(cv2.CAP_PROP_POS_MSEC,sec*1000)
    
    # Constantly reads vidcap and stores the frame in the video variable
    hasFrames, video = vidcap.read()
    if hasFrames:
        cv2.imwrite("../data/ruido_fotos/pic{}.jpg".format(str(count)), video)
    return hasFrames

def getFrame(path):
    '''
    Iterates through the frames of a video by calling the readFrame function.
    input: Path of the video we want to get the images from
    '''
    sec = 0
    frameRate = 1/30
    count=1
    success = readFrame(sec, path, count)
    while success:
        count = count + 1
        sec = sec + frameRate
        sec = round(sec, 2)
        success = readFrame(sec, path, count)

def formatImage(path):
    '''
    Converts and saves images from a folder resizing them and converting them into black and white
    input: directory or file
    '''
    filesNumber = len(os.listdir(path))

    for i in range(filesNumber):
        with open('{0}/pic{1}.jpg'.format(path,i), 'r+b') as f:
            with Image.open(f) as image:
                smaller = resizeimage.resize_cover(image, [28, 28])
                bw = smaller.convert('L')
                bw.save('../data/formatted_photos/pic{}.jpg'.format(i))


def image2array(path):
    '''
    Converts an image file into an array of (1,28,28,1) dimensions
    input: image file
    output: image array
    '''
    with open(path, 'r+b') as f:
        with Image.open(f) as image:
            cover = resizeimage.resize_cover(image, [28, 28])
            img = cover.convert('L')
            arr = np.array(img).flatten().reshape((1,28,28,1))
            return arr

def window2array(img_array):
    '''
    Converts the array that returns the sliding window to 
    dimensions (1,28,28,1) for use on the neural network
    input: image array
    output: formatted image array
    '''
    # Create Image object from array
    img = Image.fromarray(img_array)

    # Formatting to (28,28,1) object
    cover = resizeimage.resize_cover(img, [28, 28])
    cover = cover.convert('L')

    # Reshape to the neural network format
    arr = np.array(cover).flatten().reshape((1,28,28,1))
    print("Resized image ready to feed the model.")
    return arr


def plotImage(path):
    '''
    Graphically represents a black and white image
    input: image file
    '''
    arr = image2array(path)
    plt.imshow(arr.reshape((28,28)), cmap='gray')


