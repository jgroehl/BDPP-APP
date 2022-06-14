from utils.dataset import load_data
from Reference.random_forest import train_random_forest
import numpy as np

# Define paths
#DATA_PATH = "data/"
DATA_PATH = "../final_coding_Julia_newlossandclassifier/Final_Data/data"
TRAINING_PATH = DATA_PATH + "/Data_Training/"
VALIDATION_PATH = DATA_PATH + "/Data_Validation/"


# Load data
training_data, training_labels = load_data(TRAINING_PATH)
validation_data, validation_labels = load_data(VALIDATION_PATH)
print(np.shape(training_data))


train_random_forest(training_data, training_labels,
                    validation_data, validation_labels,
                    "networks/random_forest_newclassifier")