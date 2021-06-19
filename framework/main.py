import os

import tensorflow as tf
from autogluon.vision import ImagePredictor
from data.dataset.generator import generate_dataset

from configs.generator import ConfigGenerator
from data.augmentation.data_aug import DataAug
from data.features.dnn import DNNFeature
from data.features.engineer import EngineerFeature
from data.utils.data_split import DataSplit
from utils import utils
from utils.constant import Constant

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)


def main():
    yaml_file = "config.yaml"
    # parse configuration
    task_config = utils.load_yaml(yaml_file)

    # calculate similarity
    similar_datasets = DNNFeature(
        task_config,
        model_path='/home/robin/Downloads/bit_models/bit_m-r50x1_1',
        save_to_file=True
    ).calculate_similarity_topk(1)
    engineer_feature = EngineerFeature(task_config)
    model_config, framework = ConfigGenerator(
        dataset=similar_datasets[0],
        device_limit=task_config["device_limit"],
        time_limit=task_config["time_limit_sec"]).generate_config()
    # dataset spilt
    train_dataset, val_dataset, test_dataset = DataSplit(
        task_config).run_data_split()
    # dataset aug
    train_dataset = DataAug(task_config, similar_datasets,
                            engineer_feature).generate_aug_dataset(level=1)
    # generator dataset
    train_dataset, val_dataset, test_dataset = generate_dataset(
        train_dataset, val_dataset, test_dataset, framework)

    # model_predictor
    checkpoint_dir = os.path.join(os.path.dirname(__file__), "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if framework == Constant.AUTO_GLUON:
        predictor = ImagePredictor(
            path=os.path.join(task_config.get('output_path'), 'checkpoint'))
        predictor.fit(train_data=train_dataset,
                      val_data=val_dataset,
                      hyperparameters=model_config["hyperparameters"],
                      hyperparameter_tune_kwargs=model_config[
                          "hyperparameter_tune_kwargs"],
                      ngpus_per_trial=model_config["ngpus_per_trial"],
                      time_limit=model_config['time_limit'],
                      verbosity=2)
        summary = predictor.fit_summary()
        print(summary)
        # TODO
        new_config = None
        ConfigGenerator.update_config_csv(new_config)

    # calculate engineer feature
    # engineer_feature = EngineerFeature(task_config)
    # # select config
    # train_config = Selector(similar_datasets, task_config).generate_config()
    # # generate aug datset
    # train_dataset_path = DataAug(task_config.data_path, similar_datasets, engineer_feature).generate_aug_dataset(
    #     level=5)
    # # generate dataset loader
    # train_dataset_loader = AutoGluonLoader.generate_from_folder(train_dataset_path)
    # # start ensemble train
    # trails = EnsemblePredictor(train_config).fit(train_dataset_loader)
    # # start single train
    # trail = ImagePredictor(train_config).fit(train_dataset_loader)
    # generate summary


if __name__ == '__main__':
    main()
