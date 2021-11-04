import os
import argparse
import importlib
import logging
from autogluon.vision import ImagePredictor
from configuration import gluon_config_choice
from gluoncv.auto.data.dataset import ImageClassificationDataset
from utils import find_best_model


def parse_args():
    parser = argparse.ArgumentParser(description='Train a model for different kaggle competitions.')
    parser.add_argument('--data_path', type=str, default='', help='train data dir')
    parser.add_argument('--dataset', type=str, default='shopee-iet',
                        help='the kaggle competition.')
    parser.add_argument('--output_path', type=str, default='output_path/',
                        help='output path to save results.')
    parser.add_argument('--kaggle_dir', type=str, default='kaggle_dir/',
                        help='output path to save results.')
    parser.add_argument('--model_config', type=str, default='default',
                        choices=['big_models', 'best_quality', 'good_quality_fast_inference',
                                 'default_hpo', 'default', 'medium_quality_faster_inference'],
                        help='the model config for autogluon.')
    parser.add_argument('--custom', type=str, default='predict',
                        help='the name of the submission file you set.')
    parser.add_argument('--train_framework', type=str, default='autogluon', help='train framework')
    parser.add_argument('--task_name', type=str, default='', help='task name')
    parser.add_argument('--load_best_model', type=bool, default=True, help='will load the best model')
    opt = parser.parse_args()
    return opt


def predict_details(test_dataset, predictor):
    res = predictor.predict(test_dataset)
    inds, probs, value = res['id'], res['score'], res['class']
    res_prob = predictor.predict_proba(test_dataset)
    probs_all = res_prob['score']
    return inds.tolist(), probs.tolist(), probs_all.tolist(), value.tolist()


def dataset_from_folder(root, exts=('.jpg', '.jpeg', '.png')):
    items = {'image': []}
    assert isinstance(root, str)
    root = os.path.abspath(os.path.expanduser(root))
    for filename in sorted(os.listdir(root)):
        filename = os.path.join(root, filename)
        ext = os.path.splitext(filename)[1]
        if ext.lower() not in exts:
            continue
        items['image'].append(filename)
    return ImageClassificationDataset(items)


def main():
    opt = parse_args()
    out_dir = os.path.join(opt.output_path, opt.dataset, opt.model_config)
    logger = logging.getLogger('')

    dataset_path = opt.data_path

    # load the best checkpoint to evaluate
    best_checkpoint, best_config = find_best_model(output_dir=out_dir)

    # Pred results on kaggle test data
    logger.info("*" * 100)
    logger.info("Pdedict on the kaggle test data")
    try:
        importlib.import_module('gluon_task.classification.kaggle.' + opt.dataset).predict(best_checkpoint, dataset_path, opt.kaggle_dir)
    except Exception as e:
        try:
            logger.info("*" * 100)
            print(e)
            test_result = predictor.predict_proba(test_dataset)
            test_result.to_csv(os.path.join(opt.kaggle_dir, f"{opt.dataset}_original_submission.csv"))
        except Exception as e:
            logger.info("*" * 100)
            print(e)


if __name__ == '__main__':
    main()