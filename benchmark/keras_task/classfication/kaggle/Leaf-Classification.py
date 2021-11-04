import autokeras as ak
from tensorflow.keras.models import load_model

from utils import *

dataset_name = 'dog-breed-identification'
dataset_path = '/data/AutoML_compete/leaf-classification/test'
model_path = "/data/model/keras/leaf-classification/best_model"
output_csv = f"{dataset_name}-keras.csv"
input_csv = 'sample_submission.csv'

x_test = image_dataset_from_directory(dataset_path)
loaded_model = load_model(model_path, custom_objects=ak.CUSTOM_OBJECTS)
test_result = loaded_model.predict(x_test)
generate_prob_csv(test_result, dataset_path, input_csv, output_csv, )
