from data.utils.data_split import DataSplit
from expert.ocr_bert_classifier import OCRBertClassifier
from utils import utils

yaml_file = "../config.yaml"
dataset_name = "test_data"
# parse configuration
task_config = utils.load_yaml(yaml_file)

train_dataset, val_dataset, test_dataset = DataSplit(
    task_config).run_data_split()

ocr_bert = OCRBertClassifier(task_config, train_dataset, val_dataset, test_dataset)


def test_bert_train():
    ocr_bert.fit()


if __name__ == '__main__':
    test_bert_train()
