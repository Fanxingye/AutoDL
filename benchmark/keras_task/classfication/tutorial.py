import tensorflow as tf
import autokeras as ak
from sys import argv

ds_input = ak.image_dataset_from_directory(argv[1])

clf = ak.ImageClassifier(objective="val_accuracy",tuner='greedy', max_trials=5)
clf.fit(ds_input, validation_split=0.2)
