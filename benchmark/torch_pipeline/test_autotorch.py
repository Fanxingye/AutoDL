import os
import time
import argparse
import importlib
import logging
import autogluon.core as ag
from autotorch.auto import ImagePredictor
from autotorch.auto.data import TorchImageClassificationDataset


#train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders('https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
train_dataset = ImagePredictor.Dataset.from_folder('/data/AutoML_compete/CUB_200_2011/split/test')
predictor = ImagePredictor(log_dir='checkpoint')
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(train_data=train_dataset, tuning_data=train_dataset,  
            hyperparameters={'model': ag.Categorical('resnet18_v1b', 'mobilenetv3'),
                                'batch_size': ag.Categorical(1, 2),
                                'lr': ag.Real(1e-4, 1e-2, log=True), 
                                'epochs': 1, 
                                'cleanup_disk': False},
            hyperparameter_tune_kwargs={'num_trials': 1, 
                                        'max_reward': 1.0, 
                                        'searcher': 'random'},
            log_dir="checkpoint",
            nthreads_per_trial=4)  # you can trust the default config, we reduce the # epoch to save some build time