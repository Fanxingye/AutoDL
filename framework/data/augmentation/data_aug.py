import os
import random
import shutil
import string

import albumentations as A
import cv2


class DataAug:
    '''
        - dog-vs-cat
        - split
            - train
            - train_dataaug
            - test
            - val
    '''
    def __init__(self, task_config, similar_datasets, engineer_feature, aug_folder_name="train_dataaug"):
        self.task_config = task_config
        self.engineer_feature = engineer_feature
        self.dataset_path = self.task_config["data_path"]
        self.split_train_path = os.path.join(self.dataset_path, "split", "train")
        self.aug_dir = os.path.join(self.dataset_path, "split", aug_folder_name)
        self.similar_datasets = similar_datasets

    def generate_aug_dataset(self, level=2):
        # if cache or query_csv():
        #     return cache
        # return aug dataset path
        print(self.task_config)
        index = 1
        for image_label in os.listdir(self.split_train_path):
            index += 1
            image_label_dir = os.path.join(self.split_train_path, image_label)
            for image_name in os.listdir(image_label_dir):
                image = cv2.imread(os.path.join(image_label_dir, image_name))
                for x in range(level):
                    transform = A.Compose([
                        A.Rotate(limit=20, p=1),
                        A.OneOf([
                            # A.RGBShift(p=0.3),
                            A.RandomBrightnessContrast(p=0.3),
                        ], p=1),
                        A.Flip(p=0.3),
                        A.RandomResizedCrop(image.shape[0], image.shape[1], scale=(0.75, 1.0),
                                            ratio=(0.75, 1.3333333333333333), p=0.3),
                    ])
                    transformed = transform(image=image)
                    transformed_image = transformed["image"]
                    #                 transformed_image=RandomErasing(image)
                    aug_path = os.path.join(self.aug_dir, image_label)
                    if not os.path.exists(aug_path):
                        os.makedirs(aug_path)
                    save_name = f'aug{x}_' + ''.join(random.sample(string.ascii_letters + string.digits, 32)) + '.jpg'
                    cv2.imwrite(os.path.join(aug_path, save_name), transformed_image)
                shutil.copy(os.path.join(image_label_dir, image_name),
                            os.path.join(aug_path, image_name))  # copy the ori image
        return self.aug_dir

    def clear(self):
        shutil.rmtree(self.aug_dir)
        print(20*"=")
        print(self.aug_dir)
