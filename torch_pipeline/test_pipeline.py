import os
import argparse
import torch
from autotorch.auto import ImagePredictor
from autotorch.configs.generator import ConfigGenerator
from autotorch.data.generator import generate_dataset
from autotorch.features.dnn import DNNFeature
from autotorch.features.engineer import EngineerFeature
from autotorch.utils.data_split import DataSplit
from autotorch.utils import utils
from autotorch.utils.constant import Constant


def parse_args():
    parser = argparse.ArgumentParser(
        description='Train a model for different kaggle competitions.')
    parser.add_argument('--name',
                        type=str,
                        default='classification',
                        help='task name')
    parser.add_argument('--time_limit_sec',
                        type=int,
                        default=36000,
                        help='maximum time to run job')
    parser.add_argument('--data_name',
                        type=str,
                        default='A-Large-Scale-Fish-Dataset',
                        help='dataset name')
    parser.add_argument('--data_path',
                        type=str,
                        default='/data/AutoML_compete/A-Large-Scale-Fish-Dataset',
                        help='dataset path')
    parser.add_argument('--output_path',
                        type=str,
                        default='/home/yiran.wu/work_dirs/autodl_benchmark',
                        help='output path')
    parser.add_argument('--device_limit',
                        type=int,
                        default=2,
                        help='the number of devices')
    parser.add_argument('--device_type',
                        type=str,
                        default="nvidia",
                        help='the type of device')
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_args()
    task_config = vars(opt)
    checkpoint_dir = os.path.join(task_config.get('output_path'), task_config["data_name"], "checkpoint")
    if int(os.environ["LOCAL_RANK"]) == 0:
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
    # yaml_file = "config.yaml"
    # # parse configuration
    # task_config = utils.load_yaml(yaml_file)
    # calculate similarity
    similar_datasets = DNNFeature(
        task_config,
        save_to_file=True
        ).calculate_similarity_topk(1)
    # engineer_feature = EngineerFeature(task_config)
    # ef = engineer_feature.get_engineered_feature()
    print(similar_datasets)
    # print(f"similar data: {similar_datasets[0]}")
    config_generator = ConfigGenerator(
        dataset_name = task_config["data_name"],
        similar_data=similar_datasets[0],
        device_limit=task_config["device_limit"],
        time_limit=task_config["time_limit_sec"])
    model_config = config_generator.generate_config()
    print("=="*10)
    print(model_config)
    # dataset spilt
    train_dataset, val_dataset, test_dataset = DataSplit(
        task_config).run_data_split()

    # generator dataset
    train_dataset, val_dataset, test_dataset = generate_dataset(
        train_dataset, val_dataset, test_dataset)

    # model_predictor
    predictor = ImagePredictor(log_dir=checkpoint_dir)
    predictor.fit(train_data=train_dataset,
                    tuning_data=val_dataset,
                    hyperparameters=model_config["hyperparameters"],
                    hyperparameter_tune_kwargs=model_config[
                        "hyperparameter_tune_kwargs"],
                    ngpus_per_trial=model_config["ngpus_per_trial"],
                    time_limit=model_config['time_limit'])
    summary = predictor.fit_summary()
    print(summary)
    #calculate accuracy of test dataset
    test_acc, _ = predictor.evaluate(test_dataset)
    print(f"Test Accuracy: {test_acc}")
    if int(os.environ["LOCAL_RANK"]) == 0:
        config_generator.update_config_csv(checkpoint_dir)
        print("Finished!")



if __name__ == '__main__':
    main()
