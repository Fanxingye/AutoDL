import json

import requests

datasetName_Path_dict = {
     "Leaf-Classification": "/data/AutoML_compete/leaf-classification/split/",

     "A-Large-Scale-Fish-Dataset": "/data/AutoML_compete/A-Large-Scale-Fish-Dataset/split/",
    # "Store-type-recognition": "/data/AutoML_compete/Store-type-recognition/split/",
    # "DTD": "/data/AutoML_compete/dtd/split/",
    # "Weather-recognition": "/data/AutoML_compete/weather-recognitionv2/split/",
    # "Emotion-Detection": "/data/AutoML_compete/Emotion-Detection/split/",
    # "APTOS-2019-Blindness-Detection": "/data/AutoML_compete/aptos2019-blindness-detection/split/",
    # "The-Nature-Conservancy-Fisheries-Monitoring": "/data/AutoML_compete/the-nature-conservancy-fisheries-monitoring/split/",
    # "FGVC-Aircraft": "/data/AutoML_compete/fgvc-aircraft-2013b/split/",
    # "Caltech-UCSD-Birds-200-2011": "/data/AutoML_compete/CUB_200_2011/split/",
    # "Food-101": "/data/AutoML_compete/food-101/split/",
    # "Cassava-Leaf-Disease": "/data/AutoML_compete/cassava-leaf-diease/split/",
    # "National-Data-Science-Bowl": "/data/AutoML_compete/datasciencebowl/split/",
    # "Plant-Seedlings-Classification": "/data/AutoML_compete/plant-seedlings-classification/split/",

    # "Flowers-Recognition": "/data/AutoML_compete/Flowers-Recognition/split/",
    # "Dog-Breed-Identification": "/data/AutoML_compete/dog-breed-identification/split/",
    # "CINIC-10": "/data/AutoML_compete/CINIC-10/split/",
    # "MURA": "/data/AutoML_compete/MURA-v1.1/split/",
    # "Caltech-101": "/data/AutoML_compete/Caltech-101/split/",
    # "Oxford-Flower-102": "/data/AutoML_compete/oxford-102-flower-pytorch/split/",
    # "CIFAR10": "/data/AutoML_compete/cifar10/split/",
    # "CIFAR100": "/data/AutoML_compete/cifar100/split/",
    # "casting-product": "/data/AutoML_compete/casting_data/split/",
    # "sport-70": "/data/AutoML_compete/70-Sports-Image-Classification/split/",
    # "Chinese-MNIST": "/data/AutoML_compete/Chinese-MNIST/split/",
    # "dog-vs-cat": "/data/AutoML_compete/dog-vs-cat/split/",
    # "UKCarsDataset": "/data/AutoML_compete/UKCarsDataset/split/",
    # "garbage-classification": "/data/AutoML_compete/garbage_classification/split/",
    # "flying-plan": "/data/AutoML_compete/planes/split/",
    #
    #
    # "Satellite": "/data/AutoML_compete/Satellite/split/",
    # "MAMe": "/data/AutoML_compete/MAMe-dataset/split/",
    # "Road-damage": "/data/AutoML_compete/sih-road-dataset/split/",
    # "Boat-types-recognition": "/data/AutoML_compete/Boat-types-recognition/split/",
    # "Scene-Classification": "/data/AutoML_compete/Scene-Classification/split/",
    # "coins" : "/data/AutoML_compete/coins-dataset-master/split/",
    # "Bald-Classification" : "/data/AutoML_compete/Bald_Classification/split/",
     "PCB-Defects" : "/data/AutoML_compete/PCB-Defects/split/",
    # "Vietnamese-Foods" : "/data/AutoML_compete/Vietnamese-Foods/split/",
    # "yoga-pose" : "/data/AutoML_compete/yoga-pose/split/",
    # "Green-Finder" : "/data/AutoML_compete/Green-Finder/split/"
}

config_list = ['default']
# config_list = ['default']#'default_hpo', 'medium_quality_faster_inference', 'good_quality_fast_inference', 'best_quality', 'big_models', 'default'

############################################################
userName = 'yiran.wu'
# autogluon、autokeras
trainFramework = "bit"
# 数据集名称

for key in datasetName_Path_dict:
    datasetName = key
    dataSetPath = datasetName_Path_dict[key]
    for model_config in config_list:
        # datasetName = "leaf-classification"
        # datasetName =  "the-nature-conservancy-fisheries-monitoring"
        num_gpus = 4  # ngpus-per-trial
        # model_config = "default"  # ['big_models', 'best_quality', 'good_quality_fast_inference', 'default_hpo', 'default', 'medium_quality_faster_inference']
        # dataSetPath = "/data/AutoML_compete/leaf-classification/train"
        # dataSetPath = "/data/AutoML_compete/the-nature-conservancy-fisheries-monitoring/train/train"

        ############################################################

        taskName = f"{trainFramework}-{datasetName}-{model_config}"
        payloadData = {"name": taskName, "jobTrainingType": "RegularJob",
                    "engine": "apulistech/aiauto-gluon-keras:0.0.1-0.2.0-1.0.12-cuda10.1",
                    "codePath": "/home/yiran.wu",
                    "startupFile": "/data/autodl/benchmark/benchmark.py",
                    "outputPath": f"/home/{userName}/work_dirs/autodl_benchmark",
                    "datasetPath": dataSetPath, "deviceType": "nvidia_gpu_amd64_2080", "deviceNum": num_gpus,
                    "isPrivileged": False, "params":
                        {"train_framework": trainFramework,
                        "dataset": datasetName,
                        "model_config": model_config,
                        "ngpus-per-trial": str(num_gpus),
                        "task_name": taskName}, "private": True,
                    "frameworkType": "aiauto", "vcName": "platform"}
        # 请求头设置
        payloadHeader = {
            'Host': 'china-gpu02.sigsus.cn',
            'Content-Type': 'application/json',
            "Authorization": "Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1aWQiOjMwMDAxLCJ1c2VyTmFtZSI6InlpcmFuLnd1IiwiZXhwIjoxNjU3MzI3OTYzLCJpYXQiOjE2MjEzMjc5NjN9.qvf2_JvSnpeReDMSGkvpaMX0dPRobCcDdKHIkyzsLtw"
        }

        res = requests.post("http://china-gpu02.sigsus.cn/ai_arts/api/trainings/", json.dumps(payloadData),
                            headers=payloadHeader)
        print(res.text)