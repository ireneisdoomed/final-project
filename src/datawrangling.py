

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from model import create_model
from image import readFrame, getFrame, formatImage
from sklearn.metrics import accuracy_score, confusion_matrix
import pickle
import os

# Datasets
train = pd.read_csv("../data/sign_mnist/sign_mnist_train.csv")
test = pd.read_csv("../data/sign_mnist/sign_mnist_test.csv")

def whichGestures():
    '''
    Returns the most predictable gestures from a given dataset
    '''
    # Set of variables
    X_train = train.drop("label", axis=1)
    y_train = train["label"]
    X_test = test.drop("label", axis=1)
    y_test = test["label"]

    # Determining which gestures are easiest to predict using a Random Forest Classifier as a predictor
    clf = create_model().fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print("Model accuracy score:", accuracy_score(y_test, y_pred = clf.predict(X_test)))
    pickle.dump(clf, open('../output/models/RFmodel.sav', 'wb'))

    # Saving gesture confusion matrix plot
    plt.figure(figsize =(30,17))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True).set(xlabel='GROUND TRUTH', ylabel='PREDICTION', xticklabels=set(y_test), yticklabels=set(y_pred))
    plt.savefig("../output/plots/confusion_matrix.png")

def background(path='../data/ruido_videos/merge.mp4'):
    '''
    Returns dataframe with 28x28 features labeled "other" from a video
    '''
    filesNumber = len(os.listdir(path))
    getFrame(path)
    formatImage(path)

    # Dataframe with formatted images to array
    noise = pd.DataFrame()
    for i in range(filesNumber):
        with open('../data/formatted_photos/pic{}.jpeg'.format(i), 'r+b') as f:
            with Image.open(f) as image:
                arr = np.array(image)
                noise = noise.append(pd.Series(arr.flatten()), ignore_index=True)
    noise["label"] = "other"
    noise.to_csv("../data/processed/background.csv", index=False)
    return noise

def cleanedData():
    '''
    Merges the train and test datasets into one by selecting 6 gestures taking into account their predictability, 
    the number of samples and the disparity between them.
    '''
    # Change the name of numerical classes to their corresponding letter
    newLabels = {0: "A", 1: "B", 2: "C", 3: "D", 4: "E", 5: "F", 6: "G",
                 7: "H", 8: "I", 10: "K", 11: "L", 12: "M", 13: "N",
                 14: "O", 15: "P", 16: "Q", 17: "R", 18: "S",
                 19: "T", 20: "U", 21: "V", 22: "W", 23: "X",
                 24: "Y"}
    train["label"] = train["label"].map(newLabels)
    test["label"] = test["label"].map(newLabels)

    # Filter dataframes by label
    signs = ["A", "B", "F", "T", "V", "Y"]
    newTrain = pd.DataFrame()
    newTest = pd.DataFrame()
    for e in signs:
        newTrain = newTrain.append(train[train["label"] == e], ignore_index=True)
        newTest = newTest.append(test[test["label"] == e], ignore_index=True)
    print("Filtering dataframes by letter...")

    # Merge cleaned foreground dataset
    foreground = pd.concat([newTrain, newTest])
    foreground.to_csv("../data/processed/cleanedData.csv", index=False)
    print('Data saved with the cleanedData.csv filename in the data/processed directory.')

    # Merge cleaned foreground dataset with background noise
    background = pd.read_csv("../data/processed/cleanedData.csv")
    data = pd.concat([foreground, background])
    data.to_csv("../data/processed/final_data.csv", index=False)