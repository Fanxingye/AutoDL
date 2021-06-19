import os


class Constant:
    BASE_DIR = os.path.join(os.path.dirname(__file__), "../")
    ASSERT_DIR = "assert"
    BIT_FEATURES_CSV = os.path.join(BASE_DIR, ASSERT_DIR, "bit_features.csv")
    ENGINEER_FEATURES_CSV = os.path.join(BASE_DIR, ASSERT_DIR, "engineer_features.csv")
    DATASET_CONFIGURATION_CSV = os.path.join(BASE_DIR, ASSERT_DIR, "dataset_configuration.csv")
    AUTO_GLUON = "autogluon"
    PYTORCH="pytorch"
<<<<<<< HEAD
=======
    DEBUG=True
>>>>>>> fa50923b471c6816c5a13e2026dd046f097fdece


if __name__ == '__main__':
    print(Constant.BIT_FEATURES_CSV)
