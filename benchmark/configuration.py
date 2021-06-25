import os
import autogluon.core as ag
from mxnet import optimizer as optim
# from keras_block import ResNet50V1, ResNet50V2


gluon_model_list = (
            'resnet18_v1',
            'resnet34_v1',
            'resnet50_v1',
            'resnet101_v1',
            'resnet152_v1',
            'resnet18_v2',
            'resnet34_v2',
            'resnet50_v2',
            'resnet101_v2',
            'resnet152_v2',
            'resnest14',
            'resnest26',
            'resnest50',
            'resnest101',
            'resnest200',
            'resnest269',
            'se_resnet18_v1',
            'se_resnet34_v1',
            'se_resnet50_v1',
            'se_resnet101_v1',
            'se_resnet152_v1',
            'se_resnet18_v2',
            'se_resnet34_v2',
            'se_resnet50_v2',
            'se_resnet101_v2',
            'se_resnet152_v2',
            'vgg11',
            'vgg13',
            'vgg16',
            'vgg19',
            'vgg11_bn',
            'vgg13_bn',
            'vgg16_bn',
            'vgg19_bn',
            'alexnet',
            'densenet121',
            'densenet161',
            'densenet169',
            'densenet201',
            'squeezenet1.0',
            'squeezenet1.1',
            'googlenet',
            'inceptionv3',
            'xception',
            'xception71',
            'mobilenet1.0',
            'mobilenet0.75',
            'mobilenet0.5',
            'mobilenet0.25',
            'mobilenetv2_1.0',
            'mobilenetv2_0.75',
            'mobilenetv2_0.5',
            'mobilenetv2_0.25',
            'mobilenetv3_large',
            'mobilenetv3_small',
            'cifar_resnet20_v1',
            'cifar_resnet56_v1',
            'cifar_resnet110_v1',
            'cifar_resnet20_v2',
            'cifar_resnet56_v2',
            'cifar_resnet110_v2',
            'cifar_wideresnet16_10',
            'cifar_wideresnet28_10',
            'cifar_wideresnet40_8',
            'cifar_resnext29_32x4d',
            'cifar_resnext29_16x64d',
            'resnet18_v1b',
            'resnet34_v1b',
            'resnet50_v1b',
            'resnet50_v1b_gn',
            'resnet101_v1b_gn',
            'resnet101_v1b',
            'resnet152_v1b',
            'resnet50_v1c',
            'resnet101_v1c',
            'resnet152_v1c',
            'resnet50_v1d',
            'resnet101_v1d',
            'resnet152_v1d',
            'resnet50_v1e',
            'resnet101_v1e',
            'resnet152_v1e',
            'resnet50_v1s',
            'resnet101_v1s',
            'resnet152_v1s',
            'resnext50_32x4d',
            'resnext101_32x4d',
            'resnext101_64x4d',
            'resnext101e_64x4d',
            'se_resnext50_32x4d',
            'se_resnext101_32x4d',
            'se_resnext101_64x4d',
            'se_resnext101e_64x4d',
            'senet_154',
            'senet_154e',
            'darknet53',
            'nasnet_4_1056',
            'nasnet_5_1538',
            'nasnet_7_1920',
            'nasnet_6_4032',
            'residualattentionnet56',
            'residualattentionnet92',
            'residualattentionnet128',
            'residualattentionnet164',
            'residualattentionnet200',
            'residualattentionnet236',
            'residualattentionnet452',
            'cifar_residualattentionnet56',
            'cifar_residualattentionnet92',
            'cifar_residualattentionnet452',
            'resnet18_v1b_0.89',
            'resnet50_v1d_0.86',
            'resnet50_v1d_0.48',
            'resnet50_v1d_0.37',
            'resnet50_v1d_0.11',
            'resnet101_v1d_0.76',
            'resnet101_v1d_0.73',
            'mobilenet1.0_int8',
            'resnet50_v1_int8',
            'i3d_resnet50_v1_custom',
            'slowfast_4x16_resnet50_custom',
            'resnet50_v1b_custom',
            'resnet18_v1b_custom',
            'dla34',
            'hrnet_w18_c',
            'hrnet_w18_small_v1_c',
            'hrnet_w18_small_v2_c',
            'hrnet_w30_c',
            'hrnet_w32_c',
            'hrnet_w40_c',
            'hrnet_w44_c',
            'hrnet_w48_c',
            'hrnet_w64_c',
            'hrnet_w18_small_v1_s',
            'hrnet_w18_small_v2_s',
            'hrnet_w48_s')


def gluon_config_choice(dataset, model_choice="default"):

    custom_config = {
        'big_models': {
            'hyperparameters': {
                'model': ag.Categorical('resnet152_v1d', 'efficientnet_b4'),
                'lr':  ag.Categorical(0.001, 0.01, 0.1),
                'batch_size': ag.Categorical(16, 32),
                'epochs': 60,
                'early_stop_patience': -1,
                'ngpus_per_trial': 4,
                'cleanup_disk': False
            },
            'hyperparameter_tune_kwargs': {
                'num_trials': 48,
                'max_reward': 1.0,
                'searcher': 'random'
            },
            'time_limit': 3*24*3600
        },

        'search_models': {
            'hyperparameters': {
                'model': ag.Categorical('resnet152_v1d', 'efficientnet_b4'), # 'resnet152_v1d', 'efficientnet_b4', 'resnet152_v1d', 'efficientnet_b2', 
                'lr':  ag.Categorical(6e-2, 1e-1, 3e-1, 6e-1),
                'batch_size': ag.Categorical(32),
                'epochs': 50,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False
            },
            'hyperparameter_tune_kwargs': {
                'num_trials': 8,
                'max_reward': 1.0,
                'searcher': 'random'
            },
            'time_limit': 4*24*3600
        },

        'best_quality': {
            'hyperparameters': {
                'model': ag.Categorical('resnet50_v1b', 'resnet101_v1d', 'resnest200'),
                'lr': ag.Real(1e-5, 1e-2, log=True),
                'batch_size': ag.Categorical(8, 16, 32, 64, 128),
                'epochs': 120,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False
            },
            'hyperparameter_tune_kwargs': {
                'num_trials': 256,
                'searcher': 'bayesopt',
                'max_reward': 1.0,
            },
            'time_limit': 12*3600
        },

        'good_quality_fast_inference': {
            'hyperparameters': {
                'model': ag.Categorical('resnet50_v1b', 'resnet34_v1b'),
                'lr': ag.Real(1e-4, 1e-2, log=True),
                'batch_size': ag.Categorical(32, 64, 128),
                'epochs': 100,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False,
                },
            'hyperparameter_tune_kwargs': {
                'num_trials': 128,
                'max_reward': 1.0,
                'searcher': 'bayesopt',
                },
            'time_limit': 8*3600
        },

        'default_hpo': {
            'hyperparameters': {
                'model':  ag.Categorical('resnet50_v1b'),
                'lr': ag.Categorical(0.01, 0.005, 0.001, 0.02),
                'batch_size': ag.Categorical(32, 64, 128),
                'epochs': 50,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False,
            },
            'hyperparameter_tune_kwargs': {
                'num_trials':  12,
                'searcher': 'random',
                'max_reward': 1.0,
            },
            'time_limit': 16*3600
        },

        'default': {
            'hyperparameters': {
                'model': 'resnet50_v1b',
                'lr': 0.01,
                'batch_size': 64,
                'epochs': 10,
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False,
                },
            'hyperparameter_tune_kwargs': {
                'num_trials': 1,
                'max_reward': 1.0,
                },
            'time_limit': 6*3600
        },

        'medium_quality_faster_inference': {
            'hyperparameters': {
                'model': ag.Categorical('resnet18_v1b', 'mobilenetv3_small'),
                'lr': ag.Categorical(0.01, 0.005, 0.001),
                'batch_size': ag.Categorical(64, 128),
                'epochs': ag.Categorical(50, 100),
                'early_stop_patience': 10,
                'ngpus_per_trial': 4,
                'cleanup_disk': False
            },
            'hyperparameter_tune_kwargs': {
                'num_trials': 32,
                'max_reward': 1.0,
                'searcher': 'bayesopt',
                },
            'time_limit': 6*3600
        }
    }

    class NAG(optim.NAG):
        pass

    optimizer = NAG()

    if dataset == 'imagenet2012':
        config = {
                'hyperparameters': {
                    'model': ag.Categorical('resnet50_v1b', 'resnet50_v1c',
                            'resnet50_v1d', 'resnest50', 'resnet50_v1e', 'resnet50_v1s'),
                    'lr': ag.Real(1e-5, 1e-2, log=True),
                    'batch_size': ag.Categorical(16, 32, 64, 128, 256), # 8, 16, 32, 64,
                    'momentum': ag.space.Real(0.86, 0.99),
                    'wd': ag.space.Real(1e-6, 1e-2, log=True),
                    'epochs': 50,
                    'optimizer': optimizer,
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

    elif dataset == "cifar10":
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
                    'num_workers': 16,
                    'optimizer': optimizer,
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

    else:
        config = custom_config[model_choice]
    return config
