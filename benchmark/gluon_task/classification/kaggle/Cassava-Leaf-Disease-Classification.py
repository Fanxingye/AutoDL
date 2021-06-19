! pip install ../input/gluonpackages/*whl
! mkdir -p ~/.mxnet/models
! cp ../input/resnet50-v1bzip/resnet50_v1b-0ecdba34.params ~/.mxnet/models
import csv
import os

import pandas as pd
from autogluon.vision import ImagePredictor
from gluoncv.auto.data.dataset import ImageClassificationDataset


def generate_csv(images_name, preds_class, image_column_name, class_column_name, output_csv, fullname=True):
    with open(output_csv, 'w') as csvFile:
        row = [image_column_name, class_column_name]
        writer = csv.writer(csvFile)
        writer.writerow(row)
        id = 1
        for image, pred in zip(images_name, preds_class):
            if fullname:
                row = [os.path.basename(image), pred]
            else:
                row = [os.path.basename(image)[:-4], pred]
            writer = csv.writer(csvFile)
            writer.writerow(row)
            id += 1
    csvFile.close()
    print(f'generate_csv {output_csv} is done')


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


pd.set_option('display.max_columns', None)
dataset_name = 'cassava-leaf-disease-classification'
dataset_path = "../input/cassava-leaf-disease-classification"
model_path = "../input/cassgluonmodel/best_checkpoint.pkl"

test_dataset = dataset_from_folder(os.path.join(dataset_path, "test_images"))

predictor = ImagePredictor.load(model_path)
test_result = predictor.predict(test_dataset)
print(test_result)

top1 = []
for i in range(0, len(test_result), 5):
    top1.append(i)
test_result = test_result.iloc[top1]
# string image_proba -> list image_proba

generate_csv(images_name=test_result["image"].to_list(), preds_class=test_result["class"].to_list(),
             image_column_name="image_id", class_column_name="label",
             output_csv=f"submission.csv", fullname=True)
