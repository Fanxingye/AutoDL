from autogluon.vision import ImagePredictor

from .utils import *


def predict(model_path, dataset_path, output_path="./"):
    pd.set_option('display.max_columns', None)

    dataset_name = 'leaf-classification'
    predictor = ImagePredictor.load(model_path)
    test_dataset = dataset_from_folder(os.path.join(dataset_path, "test"))

    # predictor = ImagePredictor.load(model_path)
    test_result = predictor.predict_proba(test_dataset)
    # top-5 -> top-1
    # skiprows=lambda x: x > 0 and (x - 1) % 5 != 0
    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)

    # string image_proba -> list image_proba
    # images_proba = test_result['image_proba'].values.tolist()
    generate_csv_submission(dataset_path, dataset_name, "./", None, test_result.values.tolist(), None,
                            os.path.join(output_path, f"{dataset_name}_autogluon_submission"))


predict(1)
