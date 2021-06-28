import os

from autogluon.vision import ImagePredictor

from configs.generator import ConfigGenerator
from data.augmentation.data_aug import DataAug
from data.dataset.generator import generate_dataset
from data.features.dnn import DNNFeature
from data.features.engineer import EngineerFeature
from data.utils.data_split import DataSplit
from utils import utils
from utils.constant import Constant

if not Constant.DEBUG:
    import tensorflow as tf

    gpu = tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpu[0], True)


def main():
    yaml_file = "config.yaml"
    # parse configuration
    task_config = utils.load_yaml(yaml_file)
    # DEBUG
    train_dataset, val_dataset, test_dataset = DataSplit(
        task_config).run_data_split()
    framework = "autogluon"
    config_generator = ConfigGenerator(
        dataset_name="cifar10",
        device_limit=task_config["device_limit"],
        time_limit=task_config["time_limit_sec"])
    model_config, framework = config_generator.generate_config()
    data_aug = DataAug(task_config, None, None)
    # calculate similarity
    if not Constant.DEBUG:
        similar_datasets = DNNFeature(
            task_config,
            model_path='/home/yiran.wu/wyr/code/meta_features/bit_m',
            save_to_file=True
        ).calculate_similarity_topk(1)
        engineer_feature = EngineerFeature(task_config)
        config_generator = ConfigGenerator(
            dataset_name=similar_datasets[0],
            device_limit=task_config["device_limit"],
            time_limit=task_config["time_limit_sec"])
        model_config, framework = config_generator.generate_config()
        # dataset spilt
        train_dataset, val_dataset, test_dataset = DataSplit(
            task_config).run_data_split()
        # dataset aug
        data_aug = DataAug(task_config, similar_datasets,
                           engineer_feature)
        train_dataset = data_aug.generate_aug_dataset(level=1)
    # generator dataset
    train_dataset, val_dataset, test_dataset = generate_dataset(
        train_dataset, val_dataset, test_dataset, framework)

    # model_predictor
    checkpoint_dir = os.path.join(task_config.get('output_path'), "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(model_config)
    if framework == Constant.AUTO_GLUON:
        predictor = ImagePredictor(path=checkpoint_dir)
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
        # use wrapper
        config_generator.update_config_csv(checkpoint_dir)
        data_aug.clear()
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
