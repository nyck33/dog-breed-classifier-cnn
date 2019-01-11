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
from extract_bottleneck_features import *

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

#import helper functions
from app_functions import *

# Define a flask app
app = Flask(__name__)

###########################################################
#probably don't need this as the h5 model already used Resnet
#from keras.applications.resnet50 import ResNet50
# define ResNet50 model
#ResNet50_model = ResNet50(weights='imagenet')
###########################################################
# Model saved with Keras model.save()
MODEL_PATH = 'models/DogResnet50Data.weights.best.hdf5'

# Load your trained model
model = load_model(MODEL_PATH)

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
            breed_name = "Error.  Try another picture."
            return breed_name
        return None




print('Model loaded. Start serving...')    



@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = dog_breed_finder(file_path)
        return preds

    return None


if __name__ == '__main__':
    # app.run(port=5002, debug=True)

    # Serve the app with gevent
    http_server = WSGIServer(('', 5000), app)
    http_server.serve_forever()