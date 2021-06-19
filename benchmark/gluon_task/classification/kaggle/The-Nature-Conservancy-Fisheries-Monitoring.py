from autogluon.vision import ImagePredictor

from .utils import *


def predict(model_path, dataset_path, output_path="./"):
    pd.set_option('display.max_columns', None)
    dataset_name = 'the-nature-conservancy-fisheries-monitoring'

    # special for fishers
    test_dataset = dataset_from_folder(os.path.join(dataset_path, "test_stg1"))
    predictor = ImagePredictor.load(model_path)
    test_result = predictor.predict_proba(test_dataset)

    # top-5 -> top-1
    # skiprows=lambda x: x > 0 and (x - 1) % 5 != 0
    if not os.path.isdir(dataset_name):
        os.mkdir(dataset_name)

    # string image_proba -> list image_proba

    test_dataset1 = dataset_from_folder(os.path.join(dataset_path, "test_stg2"))
    test_result1 = predictor.predict(test_dataset1, with_proba=True)
    test_result.append(test_result1)
    images_proba1 = test_result1['image_proba'].values.tolist()
    csv_path = os.path.join(dataset_path, 'sample_submission_stg2.csv')
    test_path = os.path.join(dataset_path, "test_stg2")
    generate_csv_submission(dataset_path, dataset_name, "./", None, images_proba1, None,
                            os.path.join(output_path, f"{dataset_name}_autogluon_submission"), test_path, csv_path)
