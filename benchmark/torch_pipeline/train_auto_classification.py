import os
import argparse
import logging

import autogluon.core as ag
from autotorch.auto.data import TorchImageClassificationDataset
from autotorch.auto.estimators import ImageClassificationEstimator


if __name__ == '__main__':
    # user defined arguments
    parser = argparse.ArgumentParser(description='benchmark for image classification')
    parser.add_argument('--dataset', type=str, default='boat', help='dataset name')
    parser.add_argument('--num-trials', type=int, default=3, help='number of training trials')
    args = parser.parse_args()
    logging.info('user defined arguments: {}'.format(args))

    # specify hyperparameter search space
    config = {}

    # specify learning task
    train_data, valid_data, _ = TorchImageClassificationDataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    
    # fit auto estimator
    classifier = ImageClassificationEstimator(config)
    print(classifier._cfg)
    # evaluate auto estimator
    top1, top5 = classifier.fit(train_data, valid_data)
    logging.info('evaluation: top1={}, top5={}'.format(top1, top5))

    # save and load auto estimator
    classifier.save('classifier.pkl')
