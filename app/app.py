#https://www.youtube.com/watch?v=qNF1HqBvpGE
#https://github.com/gahogg/YouTube-I-mostly-use-colab-now-/tree/master/Flask%20Machine%20Learning%20Model%20on%20Heroku

#use cmd and
### py -m venv .env
### dir
### .env\scripts\activate
### cd app
### flask run
###################################### Output of LinearRegression Model with picture

import os
from ml_analyses import calculate_random_forest_score, estimate_data_with_descriptor_network, estimate_data_with_image_network
from flask import Flask, render_template, request, flash, redirect, url_for
from gupta_algo import calculate_gupta_score
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
UPLOAD_FOLDER = 'static/'
ALLOWED_EXTENSIONS = set(['png'])

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


app = Flask(__name__)
app.secret_key = 'super secret key'
app.config['SESSION_TYPE'] = 'filesystem'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/Fig.1_GraphicalAbstract4.png')
    elif request_type_str == 'POST':
        print("POST")
        print(request)
        print(request.form)
        print(request.files)

        # 1. EXTRACT DATA FROM FORM
        print("Extracting form data...")
        form = dict(request.form)
        print(form)
        input_array = np.ones((11, ))
        input_array[0] = form['mw'] if "mw" in form else -1
        input_array[1] = form['xlogP'] if "xlogP" in form else -1
        input_array[2] = form['HBD'] if "HBD" in form else -1
        input_array[3] = form['HBA'] if "HBA" in form else -1
        input_array[4] = form['RBC'] if "RBC" in form else -1
        input_array[5] = form['TPSA'] if "TPSA" in form else -1
        input_array[6] = form['HAC'] if "HAC" in form else -1
        input_array[7] = form['pka'] if "pka" in form else -1
        input_array[8] = form['LogD'] if "LogD" in form else -1
        input_array[9] = form['AR'] if "AR" in form else -1
        input_array[10] = form['ms']


        # 2. LOAD IMAGE IF POSSIBLE
        print("Uploading image data...")
        data_img = None
        if 'upload' in request.files:
            file = request.files['upload']
            if file.filename != '' and file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filename = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filename)
                # Load the image and convert it into a machine readable image with values between 0 and 1.
                im = Image.open(filename).convert("RGB")
                data_img = np.asarray(im)
                # the "min" operator will result in an image that is opnly black where there is a white background.
                data_img = 1 - (np.min(data_img, axis=2) / 255)
                os.remove(filename)


        # 3. RUN MODELS ON THE PROVIDED DATA
        print("Running models...")
        # model2 = load('static/random_forest_newclassifier.pt')
        # model3 = load('static/descriptors_classifier_newloss.pt')
        # model4 = load('static/image_classifier_best_newloss.pt')
        image_score = ""
        rf_score = calculate_random_forest_score(input_array.reshape(1, -1), 'static/random_forest_newclassifier.pt')[0]
        dl_score = estimate_data_with_descriptor_network(input_array.reshape(1, -1), "static/descriptors_classifier_newloss.pt")[0][0]
        gupta_score = calculate_gupta_score(input_array)
        if data_img is not None:
            image_score = estimate_data_with_image_network(data_img, "static/image_classifier")[0][0]

        # make_calculationRFBDPP('do_training_random_forest.py', model2, np_arr, path)
        # make_calculationDLDescript('do_training_descriptors.py', model3, np_arr, path)
        # make_calculationDLImages('do_training_images.py', model4, np_arr, path)

        return render_template('index.html', href="static/Fig.1_GraphicalAbstract4.png",
                               gupta_score=gupta_score, rf_score=rf_score,
                               dl_score=dl_score, image_score=image_score)
    else:
        raise LookupError()
