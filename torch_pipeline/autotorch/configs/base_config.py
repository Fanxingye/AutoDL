import autogluon.core as ag 


# {"type":"real","value":[1,2,3]}
def get_config(dataset):
    DATASET_CLUTSERS1 = ["cifar10", "cifar100"]
    DATASET_CLUTSERS2 = ["dog-vs-cat", "dog-breed-classicatio"]
    DATASET_CLUTSERS3 = ["UKcars", "dog-breed-classicatio", "Oxford Flower-102"]

    if dataset in DATASET_CLUTSERS1:
        config = {
                'hyperparameters': {
                    'model': ag.Categorical('cifar_resnet20_v1',
                                            'cifar_resnet56_v1',
                                            'cifar_resnet20_v2',
                                            'cifar_resnet56_v2'),
                    'lr': ag.Real(1e-2, 1e-1, log=True),
                    'batch_size': ag.Categorical(16, 32, 64),
                    'momentum': 0.9,
                    'wd': ag.space.Real(1e-5, 1e-3, log=True),
                    'epochs': 100,
                    'num_workers': 4,
                    'early_stop_patience': -1,
                    'ngpus_per_trial': 4,
                    'cleanup_disk': False
                },
                'hyperparameter_tune_kwargs': {
                    'num_trials': 1024,
                    'search_strategy': 'bayesopt'
                },
                'time_limit': 12*3600
        }
    elif dataset in DATASET_CLUTSERS2:
        config = {
                'hyperparameters': {
                    'model': ag.Categorical('resnet50_v1b', 'resnet50_v1c',
                            'resnet50_v1d', 'resnest50', 'resnet50_v1e', 'resnet50_v1s'),
                    'lr': ag.Real(1e-5, 1e-2, log=True),
                    'batch_size': ag.Categorical(16, 32, 64, 128, 256), # 8, 16, 32, 64,
                    'momentum': ag.space.Real(0.86, 0.99),
                    'wd': ag.space.Real(1e-6, 1e-2, log=True),
                    'epochs': 50,
                    'early_stop_patience': 5,
                    'ngpus_per_trial': 4,
                    'cleanup_disk': False
                },
                'hyperparameter_tune_kwargs': {
                    'num_trials': 100,
                    'search_strategy': 'bayesopt'
                },
                'time_limit': 12*3600
        }
    elif dataset in DATASET_CLUTSERS3:
        config = {
                'hyperparameters': {
                    'model': ag.Categorical('resnet50_v1b', 'resnet50_v1c',
                            'resnet50_v1d', 'resnest50', 'resnet50_v1e', 'resnet50_v1s'),
                    'lr': ag.Real(1e-5, 1e-2, log=True),
                    'batch_size': ag.Categorical(16, 32, 64, 128, 256), # 8, 16, 32, 64,
                    'momentum': ag.space.Real(0.86, 0.99),
                    'wd': ag.space.Real(1e-6, 1e-2, log=True),
                    'epochs': 50,
                    'early_stop_patience': 5,
                    'ngpus_per_trial': 4,
                    'cleanup_disk': False
                },
                'hyperparameter_tune_kwargs': {
                    'num_trials': 100,
                    'search_strategy': 'bayesopt'
                },
                'time_limit': 12*3600
        }
    else:
        config = {
            'hyperparameters': {
                'model': 'resnet18_v1b',
                'lr': 0.01,
                'batch_size': 16,
                'epochs': 1,
                'num_workers': 4,
                'early_stop_patience': 10,
                'ngpus_per_trial': 1,
                'cleanup_disk': False,
                },
            'hyperparameter_tune_kwargs': {
                'num_trials': 1,
                'max_reward': 1.0,
                },
            'time_limit': 6*3600
        }
    return config