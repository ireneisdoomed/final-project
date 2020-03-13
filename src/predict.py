

import numpy as np
import os
import keras
from keras.models import load_model
from image import formatImage, window2array


#data = pd.read_csv("../data/processed/data.csv")

def predict(img):
    '''
    Calculates the probability vector for each of the classes in the model
    and returns its index where the probability is highest.
    input: path of the image
    output: index where the probability is highest, highest probability
    '''
    # Load the trained model
    model = load_model("../output/models/final_model.json")

    # Prediction by the model
    img_predict = model.predict(img).flatten()

    return np.argmax(img_predict), max(img_predict), img_predict
    


    
'''def getKey(dictionary, value):
    
    Returns the key for a certain value from a given dictionary
    input: dictionary, value to be found
    output: key matching the desired value 
    
    for (x,y) in dictionary.items():
        if y == value:
            return x

'''