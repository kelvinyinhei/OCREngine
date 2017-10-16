


import cv2
import os 
import numpy as np
import pandas as pd
from scipy.misc import imread
import tensorflow as tf
from keras.callbacks import History
from keras.models import model_from_json
from keras.preprocessing.image import ImageDataGenerator
import keras
from keras.utils import np_utils
from keras.models import load_model #for save and reload model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.optimizers import RMSprop, SGD
import pylab
import matplotlib.pylab as plt
from numpy import argmax
import time
import json
import re




'''
Input: a list of image path 
Output: a list of predictions

what it does: Predict the letters image

function 1(): predict_image 
'''

# change the path where storing the model
json_path = "/media/SSD/SelfBuildOCR/github/Model/model.json"
h5_path = "/media/SSD/SelfBuildOCR/github/Model/model.h5"


def getJson(path):
    with open(path) as json_data:
        data = json.load(json_data)
    return data


def Predict_Image(img_path, all_img_name):


    #json_path = "/media/SSD/SelfBuildOCR/train_image/FINALFINAL/model.json"
    #h5_path = "/media/SSD/SelfBuildOCR/train_image/FINALFINAL/model.h5"


    # load json and create model
    json_file = open(json_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)


    # load weights into new model
    loaded_model.load_weights(h5_path)

    OPTIM = RMSprop()
    loaded_model.compile(loss='categorical_crossentropy', optimizer = 'adam', metrics=['accuracy'])





    wordslist = [  '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
               'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
               'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
               'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
               'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']




    label_encoder = LabelEncoder()
    encode = label_encoder.fit_transform(wordslist)

    ## encoder
    OneHE = OneHotEncoder(sparse = False)
    OHEe = encode.reshape(len(encode), 1)
    OHE = OneHE.fit_transform(OHEe)

    ## Prepare train_x and test_x data
    def convert_OHE(inputvar):

        start = 0
        end = 62
        total_letters = 1
        result = np.zeros(shape = ( 1, 62))
        inputvar = str(inputvar)
        
        for iiidx, letters in enumerate(inputvar):
            if letters not in wordslist:
                letters = ' '
            if iiidx < total_letters:
                index = wordslist.index(letters)
                var = OHE[index]
                result[start:end] = var
                start = start + 62
                end = end + 62
        
        return result


    pred_X = []
    for index, img_name in enumerate(all_img_name):
        image_path = os.path.join(img_path, img_name)
        img = imread(image_path, flatten=True)
        img = cv2.resize(img, (28,28))
        
        pred = img.reshape(1,28, 28, 1).astype('float32')
        pred /= 255.0

        prediction = loaded_model.predict(pred)
        decoder = label_encoder.inverse_transform([argmax(prediction)])
        pred_X.append(decoder[0])
        
    return pred_X



