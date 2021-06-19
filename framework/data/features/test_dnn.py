from dnn import DNNFeature
import tensorflow as tf
from types import SimpleNamespace

gpu = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpu[0], True)

config = {
  "name": "classification",
  "time_limit_sec": "36000",
  "data_name": "A-Large-Scale-Fish-Dataset1",
  "data_path": "/media/robin/DATA/datatsets/image_data/hymenoptera/images/split/train",
  "device_limit": "8",
  "device_type": "nvidia"
}

config = SimpleNamespace(**config)
dnnf = DNNFeature(config, model_path="/home/robin/jianzh/automl/autodl/framework/bit_models", save_to_file=False) 
top_3 = dnnf.calculate_similarity_topk(3)
print(top_3)