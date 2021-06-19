import json

import requests

datasetName_Path_dict = {
    # "Leaf-Classification": "/data/AutoML_compete/leaf-classification/split/train",
    # "A-Large-Scale-Fish-Dataset": "/data/AutoML_compete/A-Large-Scale-Fish-Dataset/split/train",
    # "Store-type-recognition": "/data/AutoML_compete/Store-type-recognition/split/train",
    # "DTD": "/data/AutoML_compete/dtd/split/train",
    # "Weather-recognition": "/data/AutoML_compete/weather-recognitionv2/split/train",
    # "Emotion-Detection": "/data/AutoML_compete/Emotion-Detection/split/train",
    "APTOS-2019-Blindness-Detection": "/data/AutoML_compete/aptos2019-blindness-detection/split/train",
    # "The-Nature-Conservancy-Fisheries-Monitoring": "/data/AutoML_compete/the-nature-conservancy-fisheries-monitoring/split/train",
    # "FGVC-Aircraft": "/data/AutoML_compete/fgvc-aircraft-2013b/split/train",
    # "Caltech-UCSD-Birds-200-2011": "/data/AutoML_compete/CUB_200_2011/split/train",
    # "Food-101": "/data/AutoML_compete/food-101/split/train",
    # "Cassava-Leaf-Disease": "/data/AutoML_compete/cassava-leaf-diease/split/train",
    # "National-Data-Science-Bowl": "/data/AutoML_compete/datasciencebowl/split/train",
    # "Plant-Seedlings-Classification": "/data/AutoML_compete/plant-seedlings-classification/split/train",
    # "Flowers-Recognition": "/data/AutoML_compete/Flowers-Recognition/split/train",
    "Dog-Breed-Identification": "/data/AutoML_compete/dog-breed-identification/split/train",
    # "CINIC-10": "/data/AutoML_compete/CINIC-10/split/train",
    # "MURA": "/data/AutoML_compete/MURA-v1.1/split/train",
    # "Caltech-101": "/data/AutoML_compete/Caltech-101/split/train",
    # "Oxford-Flower-102": "/data/AutoML_compete/oxford-102-flower-pytorch/split/train",
    # "CIFAR10": "/data/AutoML_compete/cifar10/split/train",
    # "CIFAR100": "/data/AutoML_compete/cifar100/split/train",
    # "casting-product": "/data/AutoML_compete/casting_data/split/train",
    # "sport-70": "/data/AutoML_compete/70-Sports-Image-Classification/split/train",
    # "Chinese-MNIST": "/data/AutoML_compete/Chinese-MNIST/split/train",
    # "dog-vs-cat": "/data/AutoML_compete/dog-vs-cat/split/train",
    # "UKCarsDataset": "/data/AutoML_compete/UKCarsDataset/split/train",
    # "garbage-classification": "/data/AutoML_compete/garbage_classification/split/train",
    # "flying-plan": "/data/AutoML_compete/planes/split/train",
    # "Satellite": "/data/AutoML_compete/Satellite/split/train",
    # "MAMe": "/data/AutoML_compete/MAMe-dataset/split/train",
    # "Road-damage": "/data/AutoML_compete/sih-road-dataset/split/train",
    # "Boat-types-recognition": "/data/AutoML_compete/Boat-types-recognition/split/train",
    # "Scene-Classification": "/data/AutoML_compete/Scene-Classification/split/train",
    # "coins": "/data/AutoML_compete/coins-dataset-master/split/train",
    # "Bald-Classification": "/data/AutoML_compete/Bald_Classification/split/train",
    # "Vietnamese-Foods": "/data/AutoML_compete/Vietnamese-Foods/split/train",
    # "yoga-pose": "/data/AutoML_compete/yoga-pose/split/train",
    # "Green-Finder": "/data/AutoML_compete/Green-Finder/split/train",
    # "MIT-Indoor-Scenes": "/data/AutoML_compete/MIT-Indoor-Scenes/split/train",
    # "Google-Scraped-Image": "/data/AutoML_compete/Google-Scraped-Image/split/train"
}

config_list = ['search_models']
# config_list = ['default']#'default_hpo', 'medium_quality_faster_inference', 'good_quality_fast_inference', 'best_quality', 'big_models', 'default'
dataaug = True
############################################################
userName = 'yiran.wu'
# autogluon、autokeras
trainFramework = "autogluon"
# 数据集名称
num_gpus = 4
for key in datasetName_Path_dict:
    datasetName = key
    dataSetPath = datasetName_Path_dict[key]
    for model_config in config_list:
        taskName = f"{trainFramework}-{datasetName}-{model_config}"
        if dataaug:
            taskName = f"{trainFramework}-{datasetName}-aug-{model_config}"
        payloadData = {"name": taskName, "jobTrainingType": "RegularJob",
                    "engine": "apulistech/aiauto-gluon-keras:0.0.1-0.2.0-1.0.12-cuda10.1",
                    "codePath": "/home/yiran.wu",
                    "startupFile": "/data/autodl/benchmark/benchmark.py",
                    "outputPath": f"/home/{userName}/work_dirs/autodl_benchmark",
                    "datasetPath": dataSetPath, "deviceType": "nvidia_gpu_amd64_2080", "deviceNum": num_gpus,
                    "isPrivileged": False, "params":
                        {"train_framework": trainFramework,
                        "dataset": datasetName,
                        "data_augmention": str(dataaug),
                        "model_config": model_config,
                        "ngpus-per-trial": str(num_gpus),
                        "report_path": "/home/yiran.wu/work_dirs/autodl_benchmark",
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
