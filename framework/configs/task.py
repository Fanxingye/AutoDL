import yaml


class TaskConfig:
    '''
    parse configuration from backend

    :param config_file:
    :return: configuration object
    '''

    def __init__(self, config_file):
        self.config_file = config_file
        # json -> object
        self.__dict__.update(yaml.load(open(config_file, "r"), Loader=yaml.FullLoader))

    def dump(self):
        # object -> dict
        return self.__dict__


if __name__ == '__main__':
    yaml_file = "../config.yaml"
    task_config = TaskConfig(yaml_file)
    print(task_config)
