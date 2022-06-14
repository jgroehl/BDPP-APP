from utils.dataset import load_data, load_image_data
from utils.deep_learning import estimate_data_with_descriptor_network
from utils.deep_learning import estimate_data_with_image_network
from utils.visualize import visualize_ROC_curves
#from utils.visualize import visualize_FeatureImportance_Descriptor_Models
#from utils.visualize import visualize_class_activation_map
from Reference.gupta_algo import calculate_gupta_score
from Reference.random_forest import calculate_random_forest_score
import numpy as np




#Load (validation / test) data
#DATA_PATH = "data/"
DATA_PATH = "../final_coding_Julia_newlossandclassifier/Final_Data/data"
VALIDATION_PATH = DATA_PATH + "/Data_Validation/"
####VALIDATION_PATH = DATA_PATH + "/Data_Test/"
###for a first trial
#VALIDATION_PATH = DATA_PATH + "/Data_Trial21012021/"
#VALIDATION_PATH = DATA_PATH + "/Data_Trial210302/"
#VALIDATION_PATH = DATA_PATH + "/Data_Trial19042021/"
#VALIDATION_PATH = DATA_PATH + "/Data_Trial05052021/"
#VALIDATION_PATH = DATA_PATH + "/Data_Trial14092021/"
#VALIDATION_PATH = DATA_PATH + "/Data_Trial10032022/"
validation_data, validation_labels = load_data(VALIDATION_PATH)
validation_data_images, _ = load_image_data(VALIDATION_PATH)

# estimate with Random Forest
estimated_labels_random_forest = calculate_random_forest_score(validation_data,
                                                              "networks/random_forest_newclassifier")
# estimate with image DL
estimated_labels_image_dl = estimate_data_with_image_network(validation_data_images,
                                                            "Networks/image_classifier_best_newloss_dataaugon")


# estimate with descriptor DL
estimated_labels_descriptor_dl = estimate_data_with_descriptor_network(validation_data,
                                                                       "networks/descriptors_classifier_newloss_dataaugon")
# estimate with GUPTA
estimated_labels_gupta =[calculate_gupta_score(datum) for datum in validation_data]

#visualize ROC curves
all_estimates = [[validation_labels, estimated_labels_descriptor_dl,"DL Descriptor Classifier", 'blue'],
                 [validation_labels, estimated_labels_image_dl,"DL Images Classifier" ,'orange'],
                 [validation_labels, estimated_labels_random_forest, "Random Forest", 'green'],
                 [validation_labels, estimated_labels_gupta,"Gupta Score", 'red']
                 ]
#visualize_ROC_curves(all_estimates)

#################
#Accuracy
estimated_labels_random_forest[estimated_labels_random_forest>=0.5]=1
estimated_labels_random_forest[estimated_labels_random_forest<0.5]=-1
#print(np.sum(estimated_labels_random_forest>=0.1)) =552
#print(np.sum(estimated_labels_random_forest<0.1)) =69
#print(np.sum(validation_labels==1))
#print(np.sum(validation_labels==-1))
TP=np.sum(estimated_labels_random_forest[validation_labels==1]==1)
TN=np.sum(estimated_labels_random_forest[validation_labels==-1]==-1)
FP=np.sum(estimated_labels_random_forest[validation_labels==1]==-1)
FN=np.sum(estimated_labels_random_forest[validation_labels==-1]==1)
print(TP+TN+FP+FN)
print("Accuracy Random Forest", (TN+TP)/(TN+TP+FN+FP))
print("Accuracy Random Forest of CNS drugs", TP/(TP+FP))
print("Accuracy Random Forest of non-CNS drugs", TN/(TN+FN))

estimated_labels_descriptor_dl[estimated_labels_descriptor_dl>=0.3]=1
estimated_labels_descriptor_dl[estimated_labels_descriptor_dl<0.3]=-1

TP=np.sum(estimated_labels_descriptor_dl[validation_labels==1]==1)
TN=np.sum(estimated_labels_descriptor_dl[validation_labels==-1]==-1)
FP=np.sum(estimated_labels_descriptor_dl[validation_labels==1]==-1)
FN=np.sum(estimated_labels_descriptor_dl[validation_labels==-1]==1)
print(TP+TN+FP+FN)
print("Accuracy Descriptor DL", (TN+TP)/(TN+TP+FN+FP))
print("Accuracy Descriptor DL of CNS drugs", TP/(TP+FP))
print("Accuracy Descriptor DL of non-CNS drugs", TN/(TN+FN))

estimated_labels_image_dl[estimated_labels_image_dl>=0.1]=1
estimated_labels_image_dl[estimated_labels_image_dl<0.1]=-1

TP=np.sum(estimated_labels_image_dl[validation_labels==1]==1)
TN=np.sum(estimated_labels_image_dl[validation_labels==-1]==-1)
FP=np.sum(estimated_labels_image_dl[validation_labels==1]==-1)
FN=np.sum(estimated_labels_image_dl[validation_labels==-1]==1)
print(TP+TN+FP+FN)
print("Accuracy Image DL", (TN+TP)/(TN+TP+FN+FP))
print("Accuracy Image DL of CNS drugs", TP/(TP+FP))
print("Accuracy Image DL of non-CNS drugs", TN/(TN+FN))

# estimated_labels_gupta[estimated_labels_gupta>=0.5]=1
# estimated_labels_gupta[estimated_labels_gupta<0.5]=-1
#
#TP=np.sum(estimated_labels_gupta[validation_labels==1]==1)
#TN=np.sum(estimated_labels_gupta[validation_labels==-1]==-1)
#FP=np.sum(estimated_labels_gupta[validation_labels==1]==-1)
#FN=np.sum(estimated_labels_gupta[validation_labels==-1]==1)
#print(TP+TN+FP+FN)
#print("Accuracy Gupta", (TN+TP)/(TN+TP+FN+FP))
#print("Accuracy Gupta", TP/(TP+FP))
#print("Accuracy Gupta", TN/(TN+FN))

#Accuracy Random Forest 0.9948186528497409
#Accuracy Descriptor DL 0.8678756476683938
#Accuracy Image DL 0.8290155440414507
#Accuracy Gupta:


####for a frist trial
#print(validation_labels)
#print(estimated_labels_random_forest)
#print(estimated_labels_image_dl)
#print(estimated_labels_gupta)
#print(estimated_labels_descriptor_dl)

#visualize Feature Importance

#visualize_FeatureImportance_Descriptor_Models([estimate_data_with_descriptor_network, hidden_layer, "DL Descriptor Classifier", 'blue'])
#visualize_FeatureImportance_Descriptor_Models([calculate_random_forest_score, tree, "Random Forest", 'green'])

#visualize CAM
#visualize_class_activation_map([validation_labels, estimated_labels_image_dl, "DL Images Classifier"])
