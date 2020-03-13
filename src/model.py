
from __future__ import print_function
import keras
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Flatten, Activation, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("../data/processed/data.csv")

def prepareData():
    '''
    Returns train and test vectors from a dataset ready to train the model
    '''
    X = data.drop("label", axis=1)
    y = data["label"]

    # Perform train/test split and one-hot-encoding of the Ground Truth
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)
    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)

    # Dataframe to numpy array
    img_rows, img_cols = 28, 28
    X_train_vector = X_train.values.reshape((X_train.shape[0], img_rows, img_cols))
    X_test_vector = X_test.values.reshape((X_test.shape[0], img_rows, img_cols))

    # Arrange data given the 'channels_last' format
    X_train_vector = X_train_vector.reshape(X_train_vector.shape[0], img_rows, img_cols, 1)
    X_test_vector = X_test_vector.reshape(X_test_vector.shape[0], img_rows, img_cols, 1)

    # Convert class vectors to class matrices
    y_train_vector = y_train.values
    y_test_vector = y_test.values

    return X_train_vector, X_test_vector, y_train_vector, y_test_vector

def create_NN():
    '''
    Definition of the neural network architecture
    output: model object
    '''
    model = Sequential()
    model.add(Conv2D(32, kernel_size=(20, 20),
                 activation='relu',
                 input_shape=(28, 28, 1)))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation='softmax'))
    return model

def train(batch_size=16, epochs=15):
    '''
    Compiling and fitting of the loaded data on the model.
    input: batch size and number of epochs parameters
    output: fit object
    '''
    model = create_NN()

    # Compilation of the model
    model.compile(loss=keras.losses.categorical_crossentropy,
                optimizer=keras.optimizers.Adadelta(),
                metrics=['accuracy'])

    # Fitting the Neural Network
    X_train_vector, X_test_vector, y_train_vector, y_test_vector = prepareData()
    training = model.fit(X_train_vector, y_train_vector,
            batch_size=batch_size,
            epochs=epochs,
            verbose=1,
            validation_data=(X_test_vector, y_test_vector))
    
    # Saving the model
    model.save("../output/models/final_model.json")
    print("Saved model to models directory.")

    return training

def create_model(max_depth=20):
    '''
    A Random Forest Classifier.
    input: The maximum depth of the tree. 20, as default
    output: Random Forest Classifier model
    '''
    model = RandomForestClassifier(max_depth, random_state=0)
    return model 

'''def load_trained_model(path):
    model = create_NN()
    return model.load_model(path)'''
