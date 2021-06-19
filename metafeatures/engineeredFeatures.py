import time
import os
import argparse

import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis, mode
from PIL import Image



'''
color mode 参考https://pillow.readthedocs.io/en/stable/handbook/concepts.html
im_per_class表示每个类别图片数量，
height表示图片高度
width表示图片宽度
area表示图片面积

mode为众数，
skew-skewness为偏度量
kurt-kurtosis为峰度量

range, std, skew, kurt均用来表示变量离散程度
'''
columns = ['name',
           'class_count',
           'image_count',
           'color_mode',

            # image per class
           'im_per_class_mean',
           'im_per_class_median',
           'im_per_class_mode',
           'im_per_class_min',
           'im_per_class_max',
           'im_per_class_range',
           'im_per_class_std',
           'im_per_class_skew',
           'im_per_class_kurt',

            # image height
           'height_mean',
           'height_median',
           'height_mode',
           'height_min',
           'height_max',
           'height_range',
           'height_std',
           'height_skew',
           'height_kurt',

            # image width
           'width_mean',
           'width_median',
           'width_mode',
           'width_min',
           'width_max',
           'width_range',
           'width_std',
           'width_skew',
           'width_kurt',

            # image area
           'area_mean',
           'area_median',
           'area_mode',
           'area_min',
           'area_max',
           'area_range',
           'area_std',
           'area_skew',
           'area_kurt']

datasetName_Path_dict = {
    "Leaf-Classification" : "/data/AutoML_compete/leaf-classification/split/train",
    "A-Large-Scale-Fish-Dataset" : "/data/AutoML_compete/A-Large-Scale-Fish-Dataset/split/train",
    "Store-type-recognition" : "/data/AutoML_compete/Store-type-recognition/split/train",
    "DTD" : "/data/AutoML_compete/dtd/split/train",
    "Weather-recognition" : "/data/AutoML_compete/weather-recognitionv2/split/train",
    "Emotion-Detection" : "/data/AutoML_compete/Emotion-Detection/split/train",
    "APTOS-2019-Blindness-Detection" : "/data/AutoML_compete/aptos2019-blindness-detection/split/train",
    "The-Nature-Conservancy-Fisheries-Monitoring" : "/data/AutoML_compete/the-nature-conservancy-fisheries-monitoring/split/train",
    "FGVC-Aircraft" : "/data/AutoML_compete/fgvc-aircraft-2013b/split/train",
    "Caltech-UCSD-Birds-200-2011" : "/data/AutoML_compete/CUB_200_2011/split/train",
    "Food-101" : "/data/AutoML_compete/food-101/split/train",
    "Cassava-Leaf-Disease" : "/data/AutoML_compete/cassava-leaf-diease/split/train",
    "National-Data-Science-Bowl" : "/data/AutoML_compete/datasciencebowl/split/train",
    "Plant-Seedlings-Classification" : "/data/AutoML_compete/plant-seedlings-classification/split/train",
    "Flowers-Recognition" : "/data/AutoML_compete/Flowers-Recognition/split/train",
    "Dog-Breed-Identification" : "/data/AutoML_compete/dog-breed-identification/split/train",
    "CINIC-10" : "/data/AutoML_compete/CINIC-10/split/train",
    "MURA" : "/data/AutoML_compete/MURA-v1.1/split/train",
    "Caltech-101" : "/data/AutoML_compete/Caltech-101/split/train",
    "Oxford-Flower-102" : "/data/AutoML_compete/oxford-102-flower-pytorch/split/train",
    "CIFAR10" : "/data/AutoML_compete/cifar10/split/train",
    "CIFAR100" : "/data/AutoML_compete/cifar100/split/train",
    "casting-product" : "/data/AutoML_compete/casting_data/split/train",
    "sport-70" : "/data/AutoML_compete/70-Sports-Image-Classification/split/train",
    "Chinese-MNIST" : "/data/AutoML_compete/Chinese-MNIST/split/train",
    "dog-vs-cat" : "/data/AutoML_compete/dog-vs-cat/split/train",
    "UKCarsDataset" : "/data/AutoML_compete/UKCarsDataset/split/train",
    "garbage-classification" : "/data/AutoML_compete/garbage_classification/split/train",
    "flying-plan" : "/data/AutoML_compete/planes/split/train",
    "Satellite" : "/data/AutoML_compete/Satellite/split/train",
    "MAMe" : "/data/AutoML_compete/MAMe-dataset/split/train",
    "Road-damage" : "/data/AutoML_compete/sih-road-dataset/split/train",
    "Boat-types-recognition" : "/data/AutoML_compete/Boat-types-recognition/split/train",
    "Scene-Classification" : "/data/AutoML_compete/Scene-Classification/split/train",
    "coins" : "/data/AutoML_compete/coins-dataset-master/split/train",
    "Bald-Classification" : "/data/AutoML_compete/Bald_Classification/split/train",
    "PCB-Defects" : "/data/AutoML_compete/PCB-Defects/split/train",
    "Vietnamese-Foods" : "/data/AutoML_compete/Vietnamese-Foods/split/train",
    "yoga-pose" : "/data/AutoML_compete/yoga-pose/split/train",
    "Green-Finder" : "/data/AutoML_compete/Green-Finder/split/train"
}

def get_list_distribution(data: np.ndarray) -> np.ndarray:
    '''
    :param data: 1d np array
    return the following statistics of input data
        mean, median, mode, min, max,
        range,std, skewness, kurtosis,
    '''
    out = np.array([np.mean(data),
                    np.median(data),
                    mode(data)[0][0],
                    np.min(data),
                    np.max(data),
                    np.max(data)-np.min(data),
                    np.std(data),
                    skew(data),
                    kurtosis(data)])
    return out



def get_data_features(ddir:str, name:str) -> np.ndarray:
    '''

    :param ddir: path to the dataset train folder
    :param name: name of the dataset
    :return: one entry of the meta features of the dataset
    '''
    os.listdir()

    imPerClass = [len(os.listdir(os.path.join(ddir, i))) for  i in os.listdir(ddir)]
    imPerClass = np.asarray(imPerClass)

    num_classes = len(os.listdir(ddir))
    num_images = np.sum(imPerClass)

    heights = []
    widths = []
    areas = []
    cmode = None

    for c in os.listdir(ddir):
        for i in os.listdir(os.path.join(ddir, c)):
            im = Image.open(os.path.join(ddir, c, i))
            size = im.size
            heights.append(size[0])
            widths.append(size[1])
            areas.append(size[0] * size[1])
            cmode = im.mode

    ipc = get_list_distribution(imPerClass)
    imh = get_list_distribution(np.asarray(heights))
    imw = get_list_distribution(np.asarray(widths))
    ima = get_list_distribution(np.asarray(areas))
    general = np.asarray([name, num_classes, num_images, cmode])
    entry = np.concatenate((general, ipc, imh, imw, ima))
    return entry


def format_csv(datasets:dict):

    entries = None
    total = len(datasets) - 1
    for i, d in enumerate(datasets.items()):
        name, ddir = d
        print(f'{i}/{total}  Process dataset {name} at {ddir}')
        entry = get_data_features(ddir, name).reshape([40,1])
        if entries is None:
            entries = entry
        else:
            entries = np.append(entries, entry, axis=1)

    print('Finished')
    print(f'entries\' shape: {entries.shape}')
    df = pd.DataFrame(entries, index=columns)
    df.to_csv('/home/yiran.wu/wyr/code/metaFeatures.csv')




if __name__ == "__main__":
    format_csv(datasetName_Path_dict)