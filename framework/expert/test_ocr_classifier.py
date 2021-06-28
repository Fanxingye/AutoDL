from ocr_bert_classifier import OCRBertClassifier

from data.utils.data_split import DataSplit
from utils import utils

yaml_file = "../config.yaml"
dataset_name = "test_data"
# parse configuration
task_config = utils.load_yaml(yaml_file)

train_dataset, val_dataset, test_dataset = DataSplit(
    task_config).run_data_split()
# train_dataset, val_dataset, test_dataset=task_config["data_path"]+"train",task_config["data_path"]+"val",task_config["data_path"]+"test"
ocr_bert = OCRBertClassifier(task_config)


def test_bert_fit():
    ocr_bert.fit(train_dataset, val_dataset, test_dataset)
def test_bert_ensemble_eval():

    ocr_bert.ensemble_eval(test_dataset)

if __name__ == '__main__':
    test_bert_ensemble_eval()

