import os
import time
import argparse
import importlib
import logging
import autogluon.core as ag
from autotorch.auto import ImagePredictor
from autotorch.proxydata import ProxyModel

train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
# train_dataset, valid_dataset, test_dataset = ImagePredictor.Dataset.from_folders("/data/AutoML_compete/A-Large-Scale-Fish-Dataset/split/")

predictor = ImagePredictor(log_dir='checkpoint')

proxmodel = ProxyModel()
proxmodel.fit(train_dataset, test_dataset)
proxy_data = proxmodel.generate_proxy_data(train_dataset, output_dir='checkpoint')
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(
    train_data=proxy_data,
    tuning_data=test_dataset,
    hyperparameters={
        'model': ag.Categorical('resnet18_v1b', 'mobilenetv3'),
        'batch_size': ag.Categorical(16),
        'lr': ag.Real(1e-4, 1e-2, log=True),
        'epochs': 1,
        'ngpus_per_trial': 1,
        'cleanup_disk': False
    },
    hyperparameter_tune_kwargs={
        'num_trials': 5,
        'max_reward': 1.0,
        'searcher': 'random'
    },
    log_dir="checkpoint",
    nthreads_per_trial=4
)  # you can trust the default config, we reduce the # epoch to save some build time

res = predictor.predict(data=proxy_data, batch_size=32)