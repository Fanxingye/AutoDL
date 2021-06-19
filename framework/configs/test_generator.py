from configs.generator import ConfigGenerator
from utils import utils

yaml_file = "../config.yaml"
dataset_name = "cifar10"
# parse configuration
task_config = utils.load_yaml(yaml_file)
cg = ConfigGenerator(dataset_name=dataset_name, device_limit=task_config["device_limit"],
                     time_limit=task_config["time_limit_sec"])


def test_csv_to_config():
    model_config = cg.generate_config()
    print(model_config)


def test_update_config():
    cg.update_config_csv("./")
    print("finish update_config_csv")


if __name__ == '__main__':
    test_csv_to_config()
    test_update_config()