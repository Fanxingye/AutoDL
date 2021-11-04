#!/bin/bash

# data split tests
python data_split.py   --data-dir /media/robin/DATA/datatsets/image_data/hymenoptera/images --split_test True
python data_split.py   --data-dir /media/robin/DATA/datatsets/image_data/hymenoptera/images

# data split
# python data_split.py --data-dir /data/AutoML_compete/leaf-classification/
# python data_split.py --data-dir /data/AutoML_compete/A-Large-Scale-Fish-Dataset/
# python data_split.py --data-dir /data/AutoML_compete/Store-type-recognition/
# python data_split.py --data-dir /data/AutoML_compete/weather-recognitionv2/
# python data_split.py --data-dir /data/AutoML_compete/Emotion-Detection/

# python data_split.py --data-dir /data/AutoML_compete/oxford-102-flower-pytorch/
# python data_split.py --data-dir /data/AutoML_compete/dtd/
python data_split.py --data-dir /data/AutoML_compete/MIT-Indoor-Scenes/