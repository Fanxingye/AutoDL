import json
import os
import numpy as np
import autogluon.core as ag
import pandas as pd
import yaml

from autotorch.utils.constant import Constant


class DefaultConfig:
    config = {
        'hyperparameters': {
            'model': ag.Categorical('resnet152_v1d'),
            'lr': ag.Categorical(1e-3, 5e-3, 1e-2, 5e-2, 1e-1),
            'batch_size': ag.Categorical(32),
            'momentum': 0.9,
            'epochs': 50,
            'early_stop_patience': 5,
            'num_workers': 8,
            'cleanup_disk': False,
            'dataset_name': 'default',
        },
        'hyperparameter_tune_kwargs': {
            # TODO need calculate
            'num_trials': 5,
            'max_reward': 1.0,
            'searcher': 'random'
        },
        'ngpus_per_trial': 4,
        'time_limit': 3 * 24 * 3600
    }


class ConfigGenerator:
    LR_RANGE = 1e-2
    EXTRA_MODELS = ["swin_base_patch4_window7_224_in22k", "pit_b_224"]

    # TODO
    # parser mongodb
    # parser csv to config
    def __init__(self, dataset_name=None, similar_data=None, time_limit=None, device_limit=1):
        self.dataset_name = dataset_name
        self.similar_data = similar_data
        self.time_limit = time_limit
        self.device_limit = device_limit

    def generate_lr_space(self,
                          base_lr=0.1,
                          real=False):
        space = []
        if real:
            #  0.01 0.1 1
            space = [base_lr * 1e-1, base_lr * 1e1]
            return ag.Real(*space), 4
        else:
            space = [
                base_lr * 0.1, base_lr, base_lr * 10
            ]
            space = [min(max(1e-4, lr), 0.9) for lr in space]
            space = list(set(space))
            return ag.Categorical(*space), len(ag.Categorical(*space))

    def generate_batch_size_space(self,
                                  batch_size=32):
        # only max value
        # TODO
        space = [batch_size * 2, batch_size, int(batch_size * 5e-1)]
        return ag.Categorical(*space), len(ag.Categorical(*space))


    def generate_model_space(self,
                             model="resnet50_v1b"):
        # TODO
        # add more model
        space = [model]
        for extra_model in self.EXTRA_MODELS:
            if extra_model not in space:
                space.append(extra_model)
                break
        return ag.Categorical(*space), len(ag.Categorical(*space))


    def generate_wd_space(self,
                          base_wd,
                          log=False):
        # TODO
        # add more model
        space = [base_wd * 1e-1, base_wd * 1e1]
        if log:
            return ag.Real(*space, log=log), 2
        else:
            return ag.Categorical(*space), len(ag.Categorical(*space))


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
        dataset_config = df.loc[df["dataset_name"] == self.similar_data]
        if dataset_config.size > 0:
            return dataset_config.to_dict(orient='records')[0]
        else:
            dataset_config = df.loc[df["dataset_name"] == "default"]
            return dataset_config.to_dict(orient='records')[0]

    SPACE_FUNCTION_DICT = {
        "lr": generate_lr_space,
        "model": generate_model_space,
        "batch_size": generate_batch_size_space,
    }

    def generate_hpo_space(self, config):
        num_trials = 1
        for key, value in config.items():
            if self.SPACE_FUNCTION_DICT.__contains__(key):
                config[key], trials = self.SPACE_FUNCTION_DICT[key](self, value)
                num_trials *= trials
        return config, num_trials

    def generate_config(self):
        '''

        :return: configuration dictionary
                 framework string
        '''
        selected_config = self.select_dataset_config()

        config = DefaultConfig.config
        # TODO
        # time estim
        selected_config, num_trials = self.generate_hpo_space(selected_config)
        config["hyperparameters"].update(selected_config)
        config["ngpus_per_trial"] = self.device_limit
        config["hyperparameter_tune_kwargs"]["num_trials"] = num_trials
        if self.time_limit:
            config["time_limit"] = self.time_limit
        if Constant.DEBUG:
            config = {'hyperparameters': {'model': 'resnet18',
                                            'lr': 0.01,
                                            'batch_size': 2,
                                            'epochs': 1, 'early_stop_patience': 1, 'num_workers': 8,
                                            'cleanup_disk': False, 'dataset_name': self.similar_data,
                                            'total_time': 3000},
                        'hyperparameter_tune_kwargs': {
                            'num_trials': 1, 'max_reward': 1.0}, 'ngpus_per_trial': 1, 'time_limit': 36000}

        return config

    def update_config_csv(self, checkpoint_dir):
        def find_best_autogluon_config(checkpoint_dir):
            _BEST_CHECKPOINT_FILE = 'best_checkpoint.pkl'
            valid_summary_file = 'fit_summary_img_cls.json'
            _BEST_CONFIG_FILE = 'config.yaml'
            best_checkpoint = ''
            best_config = ''
            best_acc = -1
            val_result = {}
            for root, dirs, files in os.walk(checkpoint_dir):
                for trial_name in dirs:
                    if trial_name.startswith('.trial_'):
                        trial_dir = os.path.join(root, trial_name)
                        try:
                            with open(os.path.join(trial_dir, valid_summary_file), 'r') as f:
                                val_result = json.load(f)
                                acc = val_result.get('valid_acc', -1)
                                if acc > best_acc and os.path.isfile(os.path.join(trial_dir, _BEST_CHECKPOINT_FILE)):
                                    best_checkpoint = os.path.join(trial_dir, _BEST_CHECKPOINT_FILE)
                                    best_config = os.path.join(trial_dir, _BEST_CONFIG_FILE)
                                    print("=" * 30, "Find a Better checkpoint : ", best_checkpoint)
                                    print("=" * 30, "Find a Better config : ", best_config)
                                    best_acc = acc
                        except Exception as e:
                            pass
            config_dict = {}
            if os.path.isfile(best_config):
                with open(best_config, 'r') as f:
                    config_yaml = yaml.load(f, Loader=yaml.FullLoader)
                    config_dict["dataset_name"] = self.dataset_name
                    config_dict["model"] = config_yaml.get('img_cls').get('model')
                    config_dict["lr"] = config_yaml.get('train').get('lr')
                    config_dict["batch_size"] = config_yaml.get('train').get('batch_size')
                    config_dict["epochs"] = config_yaml.get('train').get('epochs')
                    config_dict["early_stop_patience"] = config_yaml.get('train').get('early_stop_patience')
                    config_dict["momentum"] = config_yaml.get('train').get('momentum')
                    config_dict["weight_decay"] = config_yaml.get('train').get('weight_decay')
                    config_dict["total_time"] = val_result.get("total_time", val_result.get("time"))
                    

            return config_dict
        best_config = find_best_autogluon_config(checkpoint_dir)
        if best_config:
            config_pd = pd.read_csv(Constant.DATASET_CONFIGURATION_CSV)
            exist_rows = config_pd.loc[config_pd.dataset_name == self.dataset_name]
            print(best_config)
            if len(exist_rows) > 0:
                best_config_pd = pd.DataFrame.from_dict(best_config,orient='index').T
                config_pd[config_pd.dataset_name == self.dataset_name] = best_config_pd.iloc[0][config_pd.columns]
            else:
                config_pd = config_pd.append([best_config], ignore_index=True)
            print("finish update config ...")
            print(config_pd)
            config_pd.to_csv(Constant.DATASET_CONFIGURATION_CSV, index=False)

    def yaml_to_csv(self, yaml_path, is_update=False):
        pass


if __name__ == '__main__':
    pd.read_csv("dataset_configuration_debug.csv")
