#https://www.youtube.com/watch?v=qNF1HqBvpGE
#https://github.com/gahogg/YouTube-I-mostly-use-colab-now-/tree/master/Flask%20Machine%20Learning%20Model%20on%20Heroku

#use cmd and
### py -m venv .env
### dir
### .env\scripts\activate
### cd app
### flask run
###################################### Output of LinearRegression Model with picture

from flask import Flask, render_template, request
import numpy as np
from joblib import load
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import uuid
#from utils.dataset import load_data, load_image_data
##from utils.deep_learning import estimate_data_with_descriptor_network
##from utils.deep_learning import estimate_data_with_image_network
#from utils.visualize import visualize_ROC_curves
#from utils.visualize import visualize_FeatureImportance_Descriptor_Models
#from utils.visualize import visualize_class_activation_map
#from Reference.gupta_algo import calculate_gupta_score
#from Reference.random_forest import calculate_random_forest_score
import numpy as np

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def hello_world():
    request_type_str = request.method
    if request_type_str == 'GET':
        return render_template('index.html', href='static/Fig.1_GraphicalAbstract4.png')
    else:
        text = request.form['text']
        random_string = uuid.uuid4().hex
        path = "static/" + random_string + ".svg"
        model1 = load('gupta_algo')
        model2 = load('random_forest_newclassifier')
        model3 = load('descriptors_classifier_newloss')
        model4 = load('image_classifier_best_newloss')
        np_arr = floats_string_to_np_arr(text)
        make_calculationGupta('gupta_algo.py', model1, np_arr, path)
        make_calculationRFBDPP('do_training_random_forest.py', model2, np_arr, path)
        make_calculationDLDescript('do_training_descriptors.py', model3, np_arr, path)
        make_calculationDLImages('do_training_images.py', model4, np_arr, path)

        return render_template('index.html', href=path)

@app.route("/")
def calculate_scores():
    if request.form.validate_on_submit():
        print("VALIDATE")

def make_calculationGupta(training_data_filename, model1, new_inp_np_arr, output_file) -> np.float:
        mw = _data_array[0]
        hbd = _data_array[2]
        hba = _data_array[3]

        hac = get_heavy_atoms(_data_array[6])
        ar = get_aromatic_rings_value(_data_array[9])
        tpsa = get_tpsa(_data_array[5])
        pka = get_pka(_data_array[7])
        mwhbn = get_mwhbn(mw ** (-0.5) * (hba + hbd))

        return ar + hac + 1.5 * mwhbn + 2 * tpsa + 0.5 * pka

def make_calculationRFBDPP(training_data_filename, model2, new_inp_np_arr, output_file):
    estimated_labels_random_forest = calculate_random_forest_score(validation_data,
                                                                   "networks/random_forest_newclassifier")

def make_calculationDLDescript(training_data_filename, model3, new_inp_np_arr, output_file):
    estimated_labels_descriptor_dl = estimate_data_with_descriptor_network(validation_data,
                                                                           "networks/descriptors_classifier_newloss_dataaugon")

def make_calculationDLImages(training_data_filename, model4, new_inp_np_arr, output_file):
    estimated_labels_image_dl = estimate_data_with_image_network(validation_data_images,
                                                                 "Networks/image_classifier_best_newloss_dataaugon")

def floats_string_to_np_arr(floats_str):
    def is_float(s):
        try:
            float(s)
            return True
        except:
            return False
    floats = np.array([float(x) for x in floats_str.split(',') if is_float(x)])
    return floats.reshape(len(floats), 1)