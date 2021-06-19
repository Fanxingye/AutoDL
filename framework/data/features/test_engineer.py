from engineer import EngineerFeature
from types import SimpleNamespace

config = {
  "name": "classification",
  "time_limit_sec": "36000",
  "data_name": "A-Large-Scale-Fish-Dataset1",
  "data_path": "/media/robin/DATA/datatsets/image_data/hymenoptera/images/split/train",
  "device_limit": "8",
  "device_type": "nvidia"
}

config = SimpleNamespace(**config)
ef = EngineerFeature(config)
fea = ef._generate_feature(save_to_file=False)
print(fea)