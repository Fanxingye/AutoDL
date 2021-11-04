import json
import os

import yaml


def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as yaml_file:
        result = yaml.load(yaml_file.read(), Loader=yaml.SafeLoader)
        return result


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result


def find_best_model_loop(checkpoint_dir):
    _BEST_CHECKPOINT_FILE = 'best_checkpoint.pkl'
    valid_summary_file = 'fit_summary_img_cls.ag'
    _BEST_CONFIG_FILE = 'config.yaml'
    best_checkpoint = ''
    best_config = ''
    best_acc = -1
    result = {}
    for root, dirs, files in os.walk(checkpoint_dir):
        for trial_name in dirs:
            if trial_name.startswith('.trial_'):
                trial_dir = os.path.join(root, trial_name)
                try:
                    with open(os.path.join(trial_dir, valid_summary_file), 'r') as f:
                        result = json.load(f)
                        acc = result.get('valid_acc', -1)
                        print("=" * 30, "Check history trials results: ", trial_dir, acc)
                        if acc > best_acc and os.path.isfile(os.path.join(trial_dir, _BEST_CHECKPOINT_FILE)):
                            best_checkpoint = os.path.join(trial_dir, _BEST_CHECKPOINT_FILE)
                            best_config = os.path.join(trial_dir, _BEST_CONFIG_FILE)
                            print("=" * 30, "Find a Better checkpoint : ", best_checkpoint)
                            print("=" * 30, "Find a Better config : ", best_config)
                            best_acc = acc
                except Exception as e:
                    print(e)
    return best_checkpoint, best_config, result


def parse_config(config_file):
    with open(config_file, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    model = config.get('img_cls').get('model')
    batch_size = config.get('train').get('batch_size')
    epochs = config.get('train').get('epochs')
    learning_rate = config.get('train').get('lr')
    momentum = config.get('train').get('momentum')
    wd = config.get('train').get('wd')
    input_size = config.get('train').get('input_size')

    return model, batch_size, epochs, learning_rate, momentum, wd, input_size

if __name__ == '__main__':
    find_best_model_loop('/home/robin/jianzh/automl/autodl/benchmark/hymenoptera')
