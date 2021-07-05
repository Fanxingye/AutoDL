import os
import argparse
import logging
import pandas as pd
import autogluon.core as ag
from autotorch.auto.data import TorchImageClassificationDataset
from autotorch.auto.estimators import ImageClassificationEstimator
from autotorch.auto.task.image_classification import ImageClassification


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
    train_data, _, valid_data = TorchImageClassificationDataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
    

    # # fit auto estimator
    classifier = ImageClassificationEstimator(config)
    print(classifier._cfg)
    # # evaluate auto estimator
    classifier.fit(train_data, valid_data)
    results = classifier.evaluate(valid_data)
    print(results)

    df = classifier.predict(valid_data)
    print(df)

    x = '/media/robin/DATA/datatsets/image_data/shopee-iet/images/test/BabyPants/BabyPants_1035.jpg'
    
    out = classifier.predict_feature(x)
    print(out)
    # task
    search_args = {
        'lr': ag.Categorical(1e-3, 1e-2),
        'num_trials': 2,
        'epochs': 2,
        'num_workers': 4,
        'batch_size': ag.Categorical(4, 8),
        'search_strategy': 'random',
        'log_dir': 'checkpoint',
        'time_limits': 60*60
        }
    
    img_path = ["/data/AutoML_compete/Flowers-Recognition/split/test/daisy/10993710036_2033222c91.jpg",
                "/data/AutoML_compete/Flowers-Recognition/split/test/daisy/3475870145_685a19116d.jpg",
                "/data/AutoML_compete/Flowers-Recognition/split/test/daisy/909609509_a05ccb8127.jpg"]
    # img_path = "/data/AutoML_compete/Flowers-Recognition/split/test/daisy/10993710036_2033222c91.jpg"
    
    search_args = {'lr': ag.Categorical(1e-3, 1e-2),
                'num_trials': 1,
                'epochs': 2,
                'num_workers': 4,
                'batch_size': ag.Categorical(32, 64),
                'search_strategy': 'random',
                'time_limits': 60*60}

    task = ImageClassification(search_args)
    classifier = task.fit(train_data, valid_data)