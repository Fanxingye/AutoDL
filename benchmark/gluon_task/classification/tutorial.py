import autogluon.core as ag
from autogluon.vision import ImagePredictor

root = 'E:/model_data/dataset/storage/dog-vs-cat/dog-vs-cat-tiny'
all_data = ImagePredictor.Dataset.from_folder(root)
all_data.head()
predictor = ImagePredictor()
predictor.fit(all_data, hyperparameters={
    'model': ag.Categorical('resnet50_v1', 'resnet34_v1'),
    'lr': ag.Categorical(0.001, 0.005, 0.01, 0.02, 0.05, 0.1),
    'batch_size': ag.Categorical(16, 32),
    'epochs': 60,
    'early_stop_patience': -1,
    'ngpus_per_trial': 4,
    'cleanup_disk': False
},
hyperparameter_tune_kwargs={
  'num_trials': 48,
  'max_reward': 1.0,
  'searcher': 'random'
}, time_limit=3600 * 24)
