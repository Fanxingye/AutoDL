import albumentations as A
import cv2
import os
import math
import string
import random


class DataAug:
    def __init__(self, task_config, similar_datasets, engineer_feature, aug_folder_name="train_dataaug"):
        self.task_config = task_config
        self.engineer_feature = engineer_feature
        self.aug_folder_name = aug_folder_name
        self.similar_datasets = similar_datasets

    def generate_aug_dataset(self, level=2):
        # if cache or query_csv():
        #     return cache
        # return aug dataset path
        print(self.task_config)
        dataset_path=self.task_config["data_path"]
        path = f'{dataset_path}/split/'
        size = len(os.listdir(f'{path}/train/'))
        index = 1
        for i in os.listdir(f'{path}/train/'):
<<<<<<< HEAD
            print(f'process  {index}/{size}')
            print(f'label  {i}')

            index+=1
            for j in os.listdir(f'{path}/train/{i}'):
                image = cv2.imread(f'{path}/train/{i}/{j}')
                
=======
            index+=1
            for j in os.listdir(f'{path}/train/{i}'):
                image = cv2.imread(f'{path}/train/{i}/{j}')
>>>>>>> fa50923b471c6816c5a13e2026dd046f097fdece
                for x in range(level):
                    transform = A.Compose([
                    A.Rotate(limit=20,p=1),
                    A.OneOf([
                        #A.RGBShift(p=0.3),
                        A.RandomBrightnessContrast(p=0.3),
                    ],p=1),
                    A.Flip(p=0.3),
                    A.RandomResizedCrop(image.shape[0],image.shape[1],scale=(0.75, 1.0), ratio=(0.75, 1.3333333333333333), p=0.3),
                    ])
<<<<<<< HEAD
                    
                    transformed = transform(image=image)
                    transformed_image = transformed["image"]

    #                 transformed_image=RandomErasing(image) 
                    
=======
                    transformed = transform(image=image)
                    transformed_image = transformed["image"]
    #                 transformed_image=RandomErasing(image) 
>>>>>>> fa50923b471c6816c5a13e2026dd046f097fdece
                    save_path = f'{path}/{self.aug_folder_name}/{i}/'
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    save_name = f'aug{x}_'+''.join(random.sample(string.ascii_letters + string.digits, 32))+'.jpg'
                    cv2.imwrite(os.path.join(f'{save_path}'+save_name),transformed_image)
<<<<<<< HEAD
                print(f"{path}/{self.aug_folder_name}/{i}/{j}")

=======
>>>>>>> fa50923b471c6816c5a13e2026dd046f097fdece
                cv2.imwrite(f'{path}/{self.aug_folder_name}/{i}/{j}',image)  #copy the ori image
        return f'{path}/{self.aug_folder_name}/'
