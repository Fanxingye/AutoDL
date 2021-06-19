import os
import yaml
import json


def load_yaml(file_path):
    with open(file_path, 'r', encoding='utf-8') as yaml_file:
        result = yaml.load(yaml_file.read(), Loader=yaml.SafeLoader)
        return result


def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        result = json.load(f)
    return result
