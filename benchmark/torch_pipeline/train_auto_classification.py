import os
import argparse
import logging
import pandas as pd
import autogluon.core as ag
from autotorch.auto.data import TorchImageClassificationDataset
from autotorch.auto.estimators import ImageClassificationEstimator
from autotorch.auto.task import ImageClassification

if __name__ == '__main__':
    # specify hyperparameter search space
    config = {}
    # specify learning task
    # train_data, _, valid_data = TorchImageClassificationDataset.from_folders(
    # 'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

    train_data, valid_data, test_data = TorchImageClassificationDataset.from_folders(
        "/data/AutoML_compete/A-Large-Scale-Fish-Dataset/split/")

    # # fit auto estimator
    classifier = ImageClassificationEstimator(config)
    # print(classifier._cfg)
    # # # evaluate auto estimator
    classifier.fit(train_data, valid_data)
    # results = classifier.evaluate(valid_data)
    # print(results)

    classifier = classifier.load(
        "/data/autodl/benchmark/torch_pipeline/imageclassificationestimator-07-08-2021/best_checkpoint.pkl"
    )
    df = classifier.predict(test_data)
    # x = '/media/robin/DATA/datatsets/image_data/shopee-iet/images/test/BabyPants/BabyPants_1035.jpg'

    # out = classifier.predict_feature(x)
    # print(out)
    # # task
    # search_args = {
    #     'lr': ag.Categorical(1e-3, 1e-2),
    #     'num_trials': 2,
    #     'epochs': 2,
    #     'num_workers': 4,
    #     'batch_size': ag.Categorical(4, 8),
    #     'search_strategy': 'random',
    #     'log_dir': 'checkpoint',
    #     'time_limits': 60*60
    #     }

    # img_path = ["/data/AutoML_compete/Flowers-Recognition/split/test/daisy/10993710036_2033222c91.jpg",
    #             "/data/AutoML_compete/Flowers-Recognition/split/test/daisy/3475870145_685a19116d.jpg",
    #             "/data/AutoML_compete/Flowers-Recognition/split/test/daisy/909609509_a05ccb8127.jpg"]
    # img_path = "/data/AutoML_compete/Flowers-Recognition/split/test/daisy/10993710036_2033222c91.jpg"

    # search_args = {'lr': ag.Categorical(1e-3, 1e-2),
    #             'num_trials': 1,
    #             'epochs': 2,
    #             'num_workers': 4,
    #             'batch_size': ag.Categorical(32, 64),
    #             'search_strategy': 'random',
    #             'time_limits': 60*60}

    # # task = ImageClassification(search_args)
    # # task.load("/data/autodl/benchmark/torch_pipeline/imageclassificationestimator-07-01-2021/best_checkpoint.pkl")
    # # predictor = task.fit(train_data, valid_data)
    # df = classifier.predict_feature(img_path)
    # print(df)
