
import cv2
import matplotlib.pyplot as plt
import numpy as np
import imutils
from image import window2array
from predict import predict


def sliding_window(image):
    '''
    Draws windows on image
    input: path of the file
    output: plot with all the windows
    '''

    stepSize = 180
    (window_width, window_height) = (300,200)
    for x in range(0, image.shape[1] - window_width, stepSize):
        for y in range(0, image.shape[0] - window_height, stepSize):
            yield (x, y, image[y:y + window_height, x:x + window_width])


def pyramid(image, scale=2, minSize=(30, 30)):
    '''
    Iterator that creates the pyramids: rescales the image to a minimum size
    input: image, scale, minimum size
    output: iterator of the resized image
    '''
	# Yields the original image
	yield image

	# Keeps looping over the pyramid
	while True:
		# Computes the new dimensions of the image and resize it
		w = int(image.shape[1] / scale)
		image = imutils.resize(image, width=w)
		# If the resized image does not meet the supplied minimum
		# size, then stop constructing the pyramid
		if image.shape[0] < minSize[1] or image.shape[1] < minSize[0]:
			break
		# Yields the next image in the pyramid
		yield image


def getPyramids(path='../notebooks/pruebas/prueba0.jpg'):
    '''
    Displays the pyramids each time a key is pressed
    input: image file
    '''
    
    image = cv2.imread(path)

    for (i, resized) in enumerate(pyramid(image)):
        # Shows the resized image
        cv2.imshow("Layer {}".format(i + 1), resized)
        cv2.waitKey(0)


def detectGesture(path='../notebooks/pruebas/prueba5.jpg',*args):
    '''
    Uses the sliding window and pyramid function to detect a gesture
    input: image file or array
    '''

    if len(args)>0:
        image=args[0]
    else:
        image=cv2.imread(path)

    window_width = 300
    window_height = 200

    # The whole frame is checked before entering the image pyramid
    frame = window2array(image)
    predicted, vector = bestGesture(frame)
    if predicted:
        print("Detected gesture in full frame.")
        return     

    # loop over the image pyramid
    for pyramid_image in pyramid(image):
        # loop over the sliding window for each layer of the pyramid
        for (x, y, window) in sliding_window(pyramid_image):

            clone = pyramid_image.copy()
            cv2.rectangle(clone, (x, y), (x + window_width, y + window_height), (0, 255, 0), 2)
            cv2.imshow("Window", clone)
            cv2.waitKey(1)
            
            arr = window2array(window)
            print('Predicting window...')
            predicted, vector = bestGesture(arr)
            if predicted:
                cv2.waitKey(0)
                return
    print("No gesture is detected in the image.")
    return

def bestGesture(array_images):
    '''
    It establishes the logic in the response of the model's prediction.
    Prints response only when prediction is different from noise
    input: image array
    return: bool, probability vector
    '''

    img_predict_max_index, img_predict_max, vector  = predict(array_images)
    threshold = 0.8

    # Legible prediction
    classes = ["A", "B", "F", "T", "V", "Y", "other"]
    gesture_dict = {index:e for index,e in enumerate(classes)}

    if (img_predict_max >= threshold and img_predict_max_index != 6):
        print("The most likely gesture is: {0} with a probability of {1}.".format(gesture_dict[img_predict_max_index], img_predict_max))
        print("Gesture prediction array:",vector)
        return True,vector
    else:
        return False,vector
