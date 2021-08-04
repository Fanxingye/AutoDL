import os

from autotorch.auto import ImagePredictor
from autotorch.configs.generator import ConfigGenerator
from autotorch.data.generator import generate_dataset
from autotorch.features.dnn import DNNFeature
from autotorch.features.engineer import EngineerFeature
from autotorch.utils.data_split import DataSplit
from autotorch.utils import utils
from autotorch.utils.constant import Constant



def main():
    yaml_file = "config.yaml"
    # parse configuration
    task_config = utils.load_yaml(yaml_file)
    train_dataset, val_dataset, test_dataset = DataSplit(
        task_config).run_data_split()

    model_config = config_generator.generate_config()
    # calculate similarity
    similar_datasets = DNNFeature(
        task_config,
        save_to_file=True
        ).calculate_similarity_topk(1)
    engineer_feature = EngineerFeature(task_config)
    ef = engineer_feature.get_engineered_feature()
    config_generator = ConfigGenerator(
        dataset_name=similar_datasets[0],
        device_limit=task_config["device_limit"],
        time_limit=task_config["time_limit_sec"])
    model_config = config_generator.generate_config()
    # dataset spilt
    train_dataset, val_dataset, test_dataset = DataSplit(
        task_config).run_data_split()

    # generator dataset
    train_dataset, val_dataset, test_dataset = generate_dataset(
        train_dataset, val_dataset, test_dataset)

    # model_predictor
    checkpoint_dir = os.path.join(task_config.get('output_path'), "checkpoint")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    print(model_config)
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
    config_generator.update_config_csv(checkpoint_dir)



if __name__ == '__main__':
    main()
