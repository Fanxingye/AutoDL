import os

import autogluon.core as ag
import pandas as pd

from utils.constant import Constant


class DefaultConfig:
    autogluon = {
        'hyperparameters': {
            'model': ag.Categorical('resnet152_v1d'),
            'lr': ag.Categorical(1e-3, 5e-3, 1e-2, 5e-2, 1e-1),
            'batch_size': ag.Categorical(16, 32),
            'momentum': 0.9,
            'wd': 1e-3,
            'epochs': 50,
            'early_stop_patience': 5,
            'num_workers': 16,
            'cleanup_disk': False
        },
        'hyperparameter_tune_kwargs': {
            'num_trials': 6,
            'max_reward': 1.0,
            'searcher': 'random'
        },
        'ngpus_per_trial': 4,
        'time_limit': 3 * 24 * 3600
    }
    timm = {}


class ConfigGenerator:
    LR_RANGE = 1e-2
    EXTRA_MODELS = ["resnet50_v1b"]

    # TODO
    # parser mongodb
    # parser csv to config
    def __init__(self, dataset=None, time_limit=None, device_limit=1):
        self.dataset = dataset
        self.time_limit = time_limit
        self.device_limit = device_limit

    def generate_lr_space(self,
                          base_lr=0.1,
                          framework=Constant.AUTO_GLUON,
                          real=False):
        space = []
        if real:
            #  0.01 0.1 1
            space = [base_lr * 1e-1, base_lr * 1e1]
            if framework == Constant.AUTO_GLUON:
                return ag.Real(*space, log=True)
        else:
            # 0.01 0.05 0.1 0.5 1
            # if lr > 1.0 or lr < 1e-6:
            # 0.3 
            space = [
                base_lr * 1e-1, base_lr * 5e-1, base_lr, base_lr * 5e0,
                base_lr * 1e1
            ]
            if framework == Constant.AUTO_GLUON:
                return ag.Categorical(*space)
        return space

    def generate_batch_size_space(self,
                                  batch_size=32,
                                  framework=Constant.AUTO_GLUON):
        # only max value
        # TODO
        space = [int(batch_size * 5e-1), batch_size, batch_size * 2]
        if framework == Constant.AUTO_GLUON:
            return ag.Categorical(*space)
        return space

    def generate_model_space(self,
                             model="resnet50_v1b",
                             framework=Constant.AUTO_GLUON):
        # TODO
        # add more model
        models = [model]
        for extra_model in self.EXTRA_MODELS:
            models.append(extra_model)
        if framework == Constant.AUTO_GLUON:
            return ag.Categorical(*models)
        return models

    def generate_wd_space(self,
                          base_wd,
                          framework=Constant.AUTO_GLUON,
                          log=True):
        # TODO
        # add more model
        space = [base_wd * 1e-1, base_wd * 1e1]
        if framework == Constant.AUTO_GLUON:
            return ag.Real(*space, log=log)
        elif framework == Constant.PYTORCH:
            pass
        return space

    def select_dataset_config(self):
        '''

        :return: configuration dictionary
                 framework string
        '''
        if not os.path.isfile(Constant.DATASET_CONFIGURATION_CSV):
            raise FileNotFoundError(
                f'Cannot find csv file {Constant.DATASET_CONFIGURATION_CSV}')
        df = pd.read_csv(Constant.DATASET_CONFIGURATION_CSV)
        # TODO
        # 1 to 1 dataset_name
        dataset_config = df.loc[df["dataset_name"] == self.dataset]
        if dataset_config.size > 0:
            return dataset_config.to_dict(
                orient='records')[0], dataset_config["framework"].values[0]
        else:
            dataset_config = df.loc[df["dataset_name"] == "default"]
            return dataset_config.to_dict(
                orient='records')[0], dataset_config["framework"].values[0]

    SPACE_FUNCTION_DICT = {
        "lr": generate_lr_space,
        "model": generate_model_space,
        "wd": generate_wd_space,
        "batch_size": generate_batch_size_space,
    }

    def generate_hpo_space(self, config, framework) -> dict:
        for key, value in config.items():
            if self.SPACE_FUNCTION_DICT.__contains__(key):
                config[key] = self.SPACE_FUNCTION_DICT[key](self, value,
                                                             framework)
        return config

    def generate_config(self):
        '''

        :return: configuration dictionary
                 framework string
        '''
        selected_config, framework = self.select_dataset_config()

        if framework == Constant.AUTO_GLUON:
            config = DefaultConfig.autogluon
            # TODO
            # time estim
            selected_config = self.generate_hpo_space(selected_config,
                                                       framework)
            config["hyperparameters"].update(selected_config)
            config["ngpus_per_trial"] = self.device_limit
            if self.time_limit:
                config["time_limit"] = self.time_limit
            return config, framework

    def update_config_csv(self, config):
        config_pd = pd.read_csv(Constant.DATASET_CONFIGURATION_CSV)

    def yaml_to_csv(self, yaml_path, is_update=False):
        pass
