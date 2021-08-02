import autogluon.core as ag
from autotorch.auto import ImagePredictor
from autotorch.proxydata import ProxyModel

# train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders(
#     'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')
train_dataset, valid_dataset, _ = ImagePredictor.Dataset.from_folders(
    "/data/AutoML_compete/leafy-vegetable-pests/split/", test="None")

predictor = ImagePredictor(log_dir='checkpoint')
# predictor = predictor.load("/data/autodl/torch_pipeline/checkpoint/45357a67/.trial_0/best_checkpoint.pkl")
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(
    train_data=train_dataset,
    tuning_data=valid_dataset,
    hyperparameters={
        'model': ag.Categorical('tf_efficientnet_b4'),
        'batch_size': ag.Categorical(8),
        'lr': ag.Categorical(0.001),
        'epochs': 50,
        'cleanup_disk': False
    },
    hyperparameter_tune_kwargs={
        'num_trials': 1,
        'max_reward': 1.0,
        'searcher': 'random'
    },
    nthreads_per_trial=8,
    ngpus_per_trial=2,
    log_dir="checkpoint",
)  # you can trust the default config, we reduce the # epoch to save some build time

# res = predictor.predict(data=test_dataset, batch_size=32)

test_data = ImagePredictor.Dataset.from_folder("/data/AutoML_compete/leafy-vegetable-pests/test")
res_ = predictor.predict(data=test_data, batch_size=32)
res_.to_csv("./result.csv")
print("*"*10)
print("result saved!")
