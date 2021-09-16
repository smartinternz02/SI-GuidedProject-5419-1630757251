"""This defines the UI of the model"""

#Important Imports

import numpy as np
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer

import cv2 
from flask import Flask, render_template, request, redirect, url_for



app = Flask(__name__)  
tf.compat.v1.disable_eager_execution()

model = load_model('disaster.h5')
model._make_predict_function() 

@app.route('/')
def index():
    return render_template('home.html')


@app.route('/home', methods=['GET'])
def home():
    return render_template('home.html')


@app.route('/intro', methods=['GET'])
def intro():
    return render_template('intro.html')


@app.route('/upload', methods=['GET', 'POST'])
def upload():
    # Get a reference to webcam #0 (the default one)
    print("[INFO] starting video stream...")
    vs = cv2.VideoCapture(0)
    (W, H) = (None, None)

    # loop over frames from the video file stream
    while True:
        (grabbed, frame) = vs.read()
        if not grabbed:
            break

        if W is None or H is None:
            (H, W) = frame.shape[:2]

        output = frame.copy()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (64, 64))
        x = np.expand_dims(frame, axis=0)
        print(x.shape)
        result = np.argmax(model.predict(x), axis=-1)
        index = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
        result = str(index[result[0]])


        cv2.putText(output, "activity: {}".format(result), (10, 120), cv2.FONT_HERSHEY_PLAIN,
                    1, (0, 255, 255), 1)

        cv2.imshow("Output", output)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("p"):
            break

    print("[INFO] cleaning up...")
    vs.release()
    cv2.destroyAllWindows()
    return render_template("upload.html")


@app.route('/video', methods=['POST', 'GET'])
def video():
    if request.method == 'POST':
        uploaded_file = request.files['file1']
        if uploaded_file.filename != '':
            vid_name = str(uploaded_file.filename)
            print(vid_name + "Uploaded_Succesfully")
            uploaded_file.save(uploaded_file.filename)
            vs = cv2.VideoCapture(vid_name)
            if (vs.isOpened() == False):
                print("Error opening video stream or file")

            (W, H) = (None, None)
            while True:
                (grabbed, frame) = vs.read()
                if not grabbed:
                    break
                if W is None or H is None:
                    (H, W) = frame.shape[:2]
                output = frame.copy()
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (64, 64))
                x = np.expand_dims(frame, axis=0)
                result = np.argmax(model.predict(x), axis=-1)
                index = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
                result = str(index[result[0]])
                cv2.putText(output, "activity: {}".format(
                    result), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0, 255, 255), 1)
                cv2.imshow("Output", output)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
            print("[INFO] cleaning up...")
            vs.release()
            cv2.destroyAllWindows()
    return render_template("video.html")


@app.route('/image', methods=['POST', 'GET'])
def image():
    resulttext = ''
    if request.method == 'POST':
        uploaded_file = request.files['imgfile']
        if uploaded_file.filename != '':
            img_name = str(uploaded_file.filename)
            print(img_name + "Uploaded Succesfully")
            uploaded_file.save(uploaded_file.filename)
            from tensorflow.keras.models import load_model
            from tensorflow.keras.preprocessing import image
            # loading the model for testing
            model = load_model("disaster.h5")  
            # loading of the image
            img = image.load_img(img_name, target_size=(64, 64)) 
            x = image.img_to_array(img) 
            x = np.expand_dims(x, axis=0) 
            print(x.shape)
            # predicting the image classification
            pred = model.predict_classes(x) 
            index = ['Cyclone', 'Earthquake', 'Flood', 'Wildfire']
            result = index[pred[0]]
            resulttext = result
    return render_template('image.html', result_text=resulttext)

if __name__ == '__main__':
    app.run(debug = False, threaded = False)