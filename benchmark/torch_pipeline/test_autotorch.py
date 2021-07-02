import os
import time
import argparse
import importlib
import logging
import autogluon.core as ag
from autotorch.auto import ImagePredictor
from autotorch.auto.data import TorchImageClassificationDataset


train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

predictor = ImagePredictor()
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(train_dataset, hyperparameters={'model': ag.Categorical('resnet18_v1b', 'mobilenetv3'),'epochs': 2})  # you can trust the default config, we reduce the # epoch to save some build time
