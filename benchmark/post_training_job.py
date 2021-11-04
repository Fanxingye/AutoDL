import json

import requests

############################################################
userName = 'yiran.wu'
# autogluon、autokeras
trainFramework = "autogluon"
# 数据集名称
datasetName = "leaf-classification"
num_gpus = 1  # ngpus-per-trial
num_epochs = 1
num_trials = 1
model_config = "default"  # ['big_models', 'best_quality', 'good_quality_fast_inference', 'default_hpo', 'default', 'medium_quality_faster_inference']
dataSetPath = "/data/AutoML_compete/leaf-classification/split/train"
# dataSetPath = "/data/AutoML_compete/the-nature-conservancy-fisheries-monitoring/train/train"

############################################################

taskName = f"{trainFramework}-{datasetName}"
payloadData = {"name": taskName, "jobTrainingType": "RegularJob",
               "engine": "apulistech/aiauto-gluon-keras:0.0.1-0.2.0-1.0.12-cuda10.1",
               "codePath": "/home/yiran.wu",
               "startupFile": "/data/autodl/benchmark/benchmark.py",
               "outputPath": f"/home/{userName}/work_dirs/autodl_benchmark",
               "datasetPath": dataSetPath, "deviceType": "nvidia_gpu_amd64", "deviceNum": num_gpus,
               "isPrivileged": False, "params":
                   {"train_framework": trainFramework,
                    "dataset": datasetName,
                    "model_config": model_config,
                    "num_epochs": str(num_epochs),
                    "num_trials": str(num_trials),
                    "ngpus-per-trial": str(num_gpus),
                    "report_path": "/home/yiran.wu/work_dirs/autodl_benchmark",
                    "task_name": taskName}, "private": True,
               "frameworkType": "aiauto", "vcName": "other"}
# 请求头设置
payloadHeader = {
    'Host': 'china-gpu02.sigsus.cn',
    'Content-Type': 'application/json',
    "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjMwMDAxLCJ1c2VyTmFtZSI6InlpcmFuLnd1IiwiZXhwIjoxNjU3MzI3OTYzLCJpYXQiOjE2MjEzMjc5NjN9.qvf2_JvSnpeReDMSGkvpaMX0dPRobCcDdKHIkyzsLtw"
}

res = requests.post("http://china-gpu02.sigsus.cn/ai_arts/api/trainings/", json.dumps(payloadData),
                    headers=payloadHeader)
print(res.text)
