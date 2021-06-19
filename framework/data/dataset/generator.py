import os

from autogluon.vision import ImagePredictor

from utils.constant import Constant


def generate_dataset(train_data_dir, val_data_dir, test_data_dir, framework):
    assert os.path.exists(train_data_dir), "train data dir dose not exits"
    assert os.path.exists(val_data_dir), "val data dir dose not exits"
    assert os.path.exists(test_data_dir), "test data dir dose not exits"

    if framework == Constant.AUTO_GLUON:
        train_dataset = ImagePredictor.Dataset.from_folder(train_data_dir)
        val_dataset = ImagePredictor.Dataset.from_folder(val_data_dir)
        test_dataset = ImagePredictor.Dataset.from_folder(test_data_dir)
    elif framework == Constant.PYTORCH:
        pass

    return train_dataset, val_dataset, test_dataset
