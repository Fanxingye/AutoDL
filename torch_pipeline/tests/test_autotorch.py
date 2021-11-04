import autogluon.core as ag
import sys
sys.path.append('../')
from autotorch.auto import ImagePredictor
from autotorch.proxydata import ProxyModel

train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders(
    'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

predictor = ImagePredictor(log_dir='checkpoint')
# predictor = predictor.load("/data/autodl/torch_pipeline/checkpoint/45357a67/.trial_0/best_checkpoint.pkl")
# since the original dataset does not provide validation split, the `fit` function splits it randomly with 90/10 ratio
predictor.fit(
    train_data=train_dataset,
    tuning_data=test_dataset,
    hyperparameters={
        'model': ag.Categorical('resnet18'),
        'batch_size': ag.Categorical(32, 64),
        'lr': ag.Categorical(0.01),
        'epochs': 1,
        'cleanup_disk': False
    },
    hyperparameter_tune_kwargs={
        'num_trials': 2,
        'max_reward': 1.0,
        'searcher': 'random'
    },
    nthreads_per_trial=8,
    ngpus_per_trial=1,
)  # you can trust the default config, we reduce the # epoch to save some build time

res = predictor.predict(data=test_dataset, batch_size=32)
print(res)