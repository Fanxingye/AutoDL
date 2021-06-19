import os


class Constant:
    DEBUG = True
    BASE_DIR = os.path.join(os.path.dirname(__file__), "../")
    ASSET_DIR = "asset"
    BIT_FEATURES_CSV = os.path.join(BASE_DIR, ASSET_DIR, "bit_features.csv")
    ENGINEER_FEATURES_CSV = os.path.join(BASE_DIR, ASSET_DIR, "engineer_features.csv")
    DATASET_CONFIGURATION_CSV = os.path.join(BASE_DIR, ASSET_DIR,
                                             "dataset_configuration.csv") if DEBUG else os.path.join(BASE_DIR,
                                                                                                     ASSET_DIR,
                                                                                                     "dataset_configuration_debug.csv")
    AUTO_GLUON = "autogluon"
    PYTORCH = "pytorch"


if __name__ == '__main__':
    print(Constant.BIT_FEATURES_CSV)
