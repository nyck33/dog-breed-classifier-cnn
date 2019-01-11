#All dependencies when starting from here
from sklearn.datasets import load_files       
from keras.utils import np_utils
import numpy as np
from glob import glob
import random
import cv2                
import matplotlib.pyplot as plt   
import pickle
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image                  
from tqdm import tqdm
from keras.applications.resnet50 import preprocess_input, decode_predictions
from PIL import ImageFile                            
ImageFile.LOAD_TRUNCATED_IMAGES = True  
import h5py
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Activation
from keras.models import Sequential

from keras.callbacks import ModelCheckpoint

from keras.callbacks import ModelCheckpoint

from extract_bottleneck_features import * 

import numpy as np

from keras.layers import Dense, Flatten
from keras.models import Sequential
from keras.layers import GlobalAveragePooling2D

from keras.callbacks import ModelCheckpoint   

from extract_bottleneck_features import *

# extract pre-trained face detector
face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_alt.xml')

### returns "True" if a dog is detected in the image stored at img_path
def dog_detector(img_path):
    prediction = ResNet50_predict_labels(img_path) #need function
    return ((prediction <= 268) & (prediction >= 151)) 

# returns "True" if face is detected in image stored at img_path
def face_detector(img_path):
    img = cv2.imread(img_path) #need cv2
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray) #need function to call this
    return len(faces) > 0

#for path to one image, returns a 4D tensor suitable as input to Keras CNN (num_imgs=1, heigth=224, width=224, channels=3)
def path_to_tensor(img_path):
    # loads RGB image as PIL.Image.Image type
    img = image.load_img(img_path, target_size=(224, 224))
    # convert PIL.Image.Image type to 3D tensor with shape (224, 224, 3)
    x = image.img_to_array(img)
    # convert 3D tensor to 4D tensor with shape (1, 224, 224, 3) and return 4D tensor
    return np.expand_dims(x, axis=0)

#multiple image path array as input and returns a 4D tensor with shape (num_imgs, height=224, width=224, channels=3)
def paths_to_tensor(img_paths):
    list_of_tensors = [path_to_tensor(img_path) for img_path in tqdm(img_paths)]
    return np.vstack(list_of_tensors)

from keras.applications.resnet50 import preprocess_input, decode_predictions

def ResNet50_predict_labels(img_path):
    # returns prediction vector for image located at img_path
    img = preprocess_input(path_to_tensor(img_path))
    return np.argmax(ResNet50_model.predict(img)) #argmax is integer 
#corresponding to model's predicted object class, ie. predicted probability vector is 133 indexes?
#ie. should return a value btwn 151 and 268 inclusive if it's a dog but that's only 117

# model._make_predict_function()          # Necessary, ie. must be in the model code
def dog_breed_finder(img_path): #replaces model_predict
    bottleneck_Resnet = extract_Resnet50(path_to_tensor(img_path)) #helper function processes input
    
    predicted_vector = model.predict(bottleneck_Resnet) #dog_breed_finder does the work
    #print(predicted_vector)
    #print(dog_names[np.argmax(predicted_vector)])
    
    return dog_names[np.argmax(predicted_vector)]#returns dog name matching argmax index of 