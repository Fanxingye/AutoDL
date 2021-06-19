from configs.generator import ConfigGenerator
from utils import utils




def test_csv_to_config():
    yaml_file = "../config.yaml"
    dataset = "cifar10"
    cg = ConfigGenerator(dataset)
    # parse configuration
    task_config = utils.load_yaml(yaml_file)
    model_config = ConfigGenerator(dataset=dataset, device_limit=task_config["device_limit"],
                                   time_limit=task_config["time_limit_sec"]).generate_autogluon_config()
    print(model_config)


if __name__ == '__main__':
    test_csv_to_config()
