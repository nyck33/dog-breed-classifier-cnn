from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np

# Keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
                
from tqdm import tqdm

#needed to use the functions and Keras Resent functions
#from extract_bottleneck_features import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#import helper functions
from app_functions import *

import numpy as np

#import bottleneck features for transfer learning
bottleneck_features = np.load('bottleneck_features/DogResnet50Data.npz')
train_ResNet50 = bottleneck_features['train']
valid_ResNet50 = bottleneck_features['valid']
test_ResNet50 = bottleneck_features['test']

#print output shape of ResNet50
input_shape=train_ResNet50.shape[1:]
print (input_shape) #this prints (1,1,2048) so a vector of length 2048

### TODO: Define your architecture.
#I only need a few layers on the end
#example 1 from https://github.com/udacity/aind2-cnn/blob/master/transfer-learning/transfer_learning.ipynb
#input shape was (224,224,3) but what is it after ResNet?

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D

Resnet50_model = Sequential()
#model.add(Flatten(input_shape=(1,1,2048)))#need input shape
#model.add(Dense(133, activation='softmax'))
Resnet50_model.add(GlobalAveragePooling2D(input_shape=(1,1,2048)))#272,517 parameters without GAP
#model.add(Dense(500))
Resnet50_model.add(Dense(133, activation='softmax'))
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='Adam', 
                  metrics=['accuracy'])

Resnet50_model.summary()

### TODO: Compile the model.
Resnet50_model.compile(loss='categorical_crossentropy', optimizer='Adam', metrics=['accuracy'])

### TODO: Load the model weights with the best validation loss.
Resnet50_model.load_weights('models/DogResnet50Data.weights.best.hdf5')

### TODO: Calculate classification accuracy on the test dataset.
# get index of predicted dog breed for each image in test set
Resnet50_predictions = [np.argmax(Resnet50_model.predict(np.expand_dims(feature, axis=0))) 
                     for feature in test_ResNet50]


from extract_bottleneck_features import *


def dog_breed_finder(img_path):
    bottleneck_Resnet = extract_Resnet50(path_to_tensor(img_path))
    
    predicted_vector = Resnet50_model.predict(bottleneck_Resnet)
    #print(predicted_vector)
    #print(dog_names[np.argmax(predicted_vector)])
    
    return dog_names[np.argmax(predicted_vector)]#returns dog name matching argmax index of 
#vector

model = Resnet50_model


def closest_dog(img_path):
    is_dog = dog_detector(img_path)
    if is_dog == True:
        breed_name=dog_breed_finder(img_path)
        return breed_name
    elif is_dog == False:
        is_human = face_detector(img_path)
        if is_human ==True:
            breed_name=dog_breed_finder(img_path)#dog_breed_finder is Resnet_predict_breed
            return breed_name
        else:
            breed_name = "Error, try another picture."
            return breed_name

    return None    


#Save space on Github by saving dog names as json list             
# load list of dog names from image files
import json
import io
try:
    to_unicode = unicode
except NameError:
    to_unicode = str

#dog_names = [item[10:-1] for item in sorted(glob("train/*/"))]

#print(dog_names)

# Write JSON file
#with io.open('dog_names.json', 'w', encoding='utf8') as outfile:
 #   str_ = json.dumps(dog_names,
  #                    indent=4, sort_keys=True,
   #                   separators=(',', ': '), ensure_ascii=False)
    #outfile.write(to_unicode(str_))

# Read JSON file
with open('dog_names.json') as data_file:
    breed_names = json.load(data_file)

print(breed_names)

dog_names = breed_names

model.summary()

model.save('models/my_model.h5')