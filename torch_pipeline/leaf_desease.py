import autogluon.core as ag
from autotorch.auto import ImagePredictor
from autotorch.proxydata import ProxyModel


train_dataset = ImagePredictor.Dataset.from_folder("/media/robin/DATA/datatsets/image_data/leaf_diseases/train")

predictor = ImagePredictor(log_dir='checkpoint')
# predictor = predictor.load("/data/autodl/torch_pipeline/checkpoint/bddb5b08/.trial_0/best_checkpoint.pkl")
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(
    train_data=train_dataset,
    tuning_data=None,
    hyperparameters={
        'model': ag.Categorical('resnet18'),
        'batch_size': ag.Categorical(32),
        'lr': ag.Categorical(1e-2),
        'epochs': 1,
        'data_augment': 'autoaugment',
        'mixup': True,
        'log_interval': 10,
        'cleanup_disk': False
    },
    hyperparameter_tune_kwargs={
        'num_trials': 1,
        'max_reward': 1.0,
        'searcher': 'random'
    },
    log_dir="checkpoint",
    nthreads_per_trial=0,
    ngpus_per_trial=1,
)  # you can trust the default config, we reduce the # epoch to save some build time

# res = predictor.predict(data=test_dataset, batch_size=32)

test_data = ImagePredictor.Dataset.from_folder("/media/robin/DATA/datatsets/image_data/leaf_diseases/test")
res_ = predictor.predict(data=test_data, batch_size=32)
res_.to_csv("checkpoint/result.csv")