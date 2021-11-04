from autogluon.vision import ImagePredictor

from .utils import *


def predict(model_path, dataset_path, output_path="./"):
    pd.set_option('display.max_columns', None)
    dataset_name = 'oxford-flower-102'


    test_dataset = dataset_from_folder(os.path.join(dataset_path, "test"))

    # oxford
    predictor = ImagePredictor.load(model_path)
    test_result = predictor.predict(test_dataset)

    labels = []
    for folder in sorted(os.listdir(os.path.join(dataset_path, "train"))):
        print(folder)
        labels.append(folder)
    # top-5 -> top-1
    # skiprows=lambda x: x > 0 and (x - 1) % 5 != 0
    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)

    # oxford
    # string image_proba -> list image_proba
    classes_name = test_result["class"].map(lambda x: labels[x]).values.tolist()
    generate_csv(images_name=test_result["image"].to_list(), preds_class=classes_name,
                 image_column_name="file_name", class_column_name="id",
                 output_csv=os.path.join(output_path, f"{dataset_name}_autogluon_submission"))
