from autotorch.auto.data import TorchImageClassificationDataset
from autotorch.auto.estimators import ImageClassificationEstimator

if __name__ == '__main__':
    # specify hyperparameter search space
    config = {}
    # specify learning task
    # train_data, _, valid_data = TorchImageClassificationDataset.from_folders(
    # 'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

    train_data, valid_data, test_data = TorchImageClassificationDataset.from_folders(
        "/data/AutoML_compete/leafy-vegetable-pests/split/")
    # # fit auto estimator
    classifier = ImageClassificationEstimator(config)
    # evaluate auto estimator
    classifier.fit(train_data, valid_data)
    results = classifier.evaluate(valid_data)
    print(results)
    classifier = classifier.load(
        "/data/autodl/benchmark/torch_pipeline/imageclassificationestimator-07-08-2021/best_checkpoint.pkl"
    )
    df = classifier.predict(test_data)
    print(df)