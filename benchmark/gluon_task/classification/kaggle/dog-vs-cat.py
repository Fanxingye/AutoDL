from autogluon.vision import ImagePredictor

from .utils import *

pd.set_option('display.max_columns', None)


def predict(model_path, dataset_path, output_path="./"):
    dataset_name = 'dog-vs-cat'
    test_dataset = dataset_from_folder(os.path.join(dataset_path, "test"))

    predictor = ImagePredictor.load(model_path)
    test_result = predictor.predict(test_dataset)
    print(test_result)
    # top-5 -> top-1
    # skiprows=lambda x: x > 0 and (x - 1) % 5 != 0
    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)

    # string image_proba -> list image_proba
    # images_proba = test_result['image_proba'].values.tolist()

    generate_csv(images_name=test_result["image"].to_list(), preds_class=test_result["class"].to_list(),
                 image_column_name="id", class_column_name="label",
                 output_csv=os.path.join(output_path, f"{dataset_name}_autogluon_submission"), fullname=False)
