from autotorch.auto.data import TorchImageClassificationDataset
from autotorch.auto.estimators import ImageClassificationEstimator

if __name__ == '__main__':
    # specify hyperparameter search space
    config = {}
    # specify learning task
    train_data, _, valid_data = TorchImageClassificationDataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

    classifier = ImageClassificationEstimator(config)
    # evaluate auto estimator
    classifier.fit(train_data, valid_data)
    results = classifier.evaluate(valid_data)
    print(results)