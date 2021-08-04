from autotorch.features.dnn import DNNFeature
from types import SimpleNamespace

config = {
  "name": "classification",
  "time_limit_sec": "36000",
  "data_name": "A-Large-Scale-Fish-Dataset2",
  "data_path": "/data/AutoML_compete/A-Large-Scale-Fish-Dataset/split/train",
  "device_limit": "8",
  "device_type": "nvidia"
}

dnnf = DNNFeature(config, save_to_file=False) 
top_3 = dnnf.calculate_similarity_topk(3)
print(top_3)