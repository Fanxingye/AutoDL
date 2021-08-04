from autotorch.features.dnn import DNNFeature

config = {
  "name": "classification",
  "time_limit_sec": "36000",
  "data_name": "A-Large-Scale-Fish-Dataset2",
  "data_path": "/data/AutoML_compete/A-Large-Scale-Fish-Dataset/split/train",
  "device_limit": "8",
  "device_type": "nvidia"
}

config = {
  "name": "classification",
  "time_limit_sec": "36000",
  "data_name": "A-Large-Scale-Fish-Dataset1",
  "data_path": "/media/robin/DATA/datatsets/image_data/hymenoptera/split/train",
  "device_limit": "8",
  "device_type": "nvidia"
}


dnnf = DNNFeature(config, save_to_file=False) 
top_3 = dnnf.calculate_similarity_topk(3)
print(top_3)