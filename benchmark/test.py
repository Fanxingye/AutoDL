import importlib

# from gluon_task.classification.kaggle import LeafClassification

dataset = "Leaf-Classification"

predictor = importlib.import_module('gluon_task.classification.kaggle.' + dataset)
predictor_result = predictor.predict("1", "1")
