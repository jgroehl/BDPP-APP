from utils.dataset import load_data
from utils.deep_learning import train_descriptor_classifier
import numpy as np
import torch

# Define paths
#DATA_PATH = "data/"
DATA_PATH = "../final_coding_Julia_newlossandclassifier/Final_Data/data"
TRAINING_PATH = DATA_PATH + "/Data_Training/"
VALIDATION_PATH = DATA_PATH + "/Data_Validation/"

##Reproducibility
RANDOM_SEED = 16110708
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# Load data
training_data, training_labels = load_data(TRAINING_PATH)
validation_data, validation_labels = load_data(VALIDATION_PATH)
print(np.shape(training_data))

# Train classifier
train_descriptor_classifier(training_data, training_labels,
                            validation_data, validation_labels,
                            "networks/descriptors_classifier_newloss_dataaugon")

