from utils.dataset import load_image_data
from utils.deep_learning import train_image_classifier
import numpy as np


# Define paths
#DATA_PATH = "data/"
DATA_PATH = "../final_coding_Julia_newlossandclassifier/Final_Data/data"
TRAINING_PATH = DATA_PATH + "/Data_Training/"
EXTRA_TRAINING_PATH = DATA_PATH + "/Data_final_unclear/"
VALIDATION_PATH = DATA_PATH + "/Data_Validation/"


# Load data
training_data, training_labels = load_image_data(TRAINING_PATH)
extra_training_data, extra_training_labels = load_image_data(EXTRA_TRAINING_PATH)
training_data = np.vstack([training_data, extra_training_data])
training_labels = np.hstack([training_labels, extra_training_labels])
validation_data, validation_labels = load_image_data(VALIDATION_PATH)


# Train classifier
train_image_classifier(training_data, training_labels,
                       validation_data, validation_labels,
                       "Networks/image_classifier_best_newloss_dataaugon")

