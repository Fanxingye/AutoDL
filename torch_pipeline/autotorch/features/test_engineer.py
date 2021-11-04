from engineer import EngineerFeature
from types import SimpleNamespace
import os
BASE_DIR = os.path.join(os.path.dirname(__file__), "../")
ASSET_DIR = os.path.join(BASE_DIR, "asset")
BIT_FEATURES_CSV = os.path.join(ASSET_DIR, "bit_features.csv")
ENGINEER_FEATURES_CSV = "/home/yiran.wu/wyr/code/autodl/framework/asset/engineer_features.csv"

config = {
  "name": "classification",
  "time_limit_sec": "36000",
  "data_name": "yoga",
  "data_path": "/data/AutoML_compete/yoga-pose/split/train",
  "device_limit": "8",
  "device_type": "nvidia"
}



# config = SimpleNamespace(**config)
ef = EngineerFeature(config, ENGINEER_FEATURES_CSV)

print(ef.contain_faces())
print(ef.contain_poses())
print(ef.contain_chars())