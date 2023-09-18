from __future__ import division, print_function
# coding=utf-8
import sys
import os
import glob
import re
import numpy as np
import tensorflow as tf
from PIL import Image
from io import BytesIO
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
# Keras
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
#from gevent.pywsgi import WSGIServer

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
# MODEL_PATH ='D:/Data Science Datasets/Potato dataset/models/potatoes.h5'
#
# # Load your trained model
# model = load_model(MODEL_PATH)
#
# class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

MODEL = tf.keras.models.load_model("D:/Data Science Datasets/Potato dataset/models/1")

CLASS_NAMES = ["Early Blight", "Late Blight", "Healthy"]

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img.numpy())
    img_array = tf.expand_dims(img_array, 0)

    predictions = model.predict(img_array)

    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, confidence

# predicted_class, confidence = predict(model, images[i].numpy())

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
        file_path = 'D:/Data Science Datasets/Potato dataset/archive/Plant_Village/Potato___Early_blight/'+secure_filename(f.filename)
        f.save(file_path)
        #
        # # Make prediction
        # predicted_class, confidence = predict(model, file_path)
        # # preds = model_predict(file_path, model)
        # result=predicted_class
        print(file_path)
        with open(file_path, "rb") as file:
            image_data = file.read()
            image = read_file_as_image(image_data)
        # image = read_file_as_image(file_path.read())
        img_batch = np.expand_dims(image, 0)

        predictions = MODEL.predict(img_batch)

        predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
        confidence = np.max(predictions[0])
        print('Confidence:',confidence)
        return predicted_class
    return None

if __name__ == '__main__':
    app.run(port=5001,debug=True)
