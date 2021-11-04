import autogluon.core as ag
from autotorch.auto import ImagePredictor
from autotorch.proxydata import ProxyModel

# train_dataset, _, valid_dataset = ImagePredictor.Dataset.from_folders(
#     'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
# train_dataset, valid_dataset, test_dataset = ImagePredictor.Dataset.from_folders(
#     "/data/AutoML_compete/leafy-vegetable-pests/split/", test="None")

train_dataset, valid_dataset, _ = ImagePredictor.Dataset.from_folders(
    "/data/AutoML_compete/advertising-image-material/split/", test="None")
# proxy_model = ProxyModel()
# proxy_model.fit(train_dataset, valid_dataset)
# train_dataset = proxy_model.generate_proxy_data(train_dataset)
predictor = ImagePredictor(log_dir='checkpoint')
# predictor = predictor.load("/data/autodl/torch_pipeline/checkpoint/45357a67/.trial_0/best_checkpoint.pkl")
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(
    train_data=train_dataset,
    tuning_data=valid_dataset,
    hyperparameters={
        'model': ag.Categorical('swin_base_patch4_window12_384'),
        'batch_size': ag.Categorical(8),
        'lr': ag.Categorical(0.001, 0.005, 0.0005, 0.0001),
        'epochs': 50,
        'cleanup_disk': False
    },
    hyperparameter_tune_kwargs={
        'num_trials': 4,
        'max_reward': 1.0,
        'searcher': 'random'
    },
    nthreads_per_trial=8,
    ngpus_per_trial=1,
    holdout_frac=0.1
)  # you can trust the default config, we reduce the # epoch to save some build time

# test_acc, _ = predictor.evaluate(test_dataset)
# print(f"Test Accuracy: {test_acc}")
# res = predictor.predict(data=test_dataset, batch_size=32)

# test_data = ImagePredictor.Dataset.from_folder("/data/AutoML_compete/leafy-vegetable-pests/test")
# res_ = predictor.predict(data=test_data, batch_size=32)
# res_.to_csv("./result.csv")
# print("*"*10)
# print("result saved!")
