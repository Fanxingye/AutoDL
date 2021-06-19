# -*- coding: utf-8 -*-
"""
Created on Wed May 26 17:19:58 2021

@author: DELL
"""
import albumentations as A
import cv2
import sys
import os
import re
import math
import numpy
import shutil
import string
import random
import argparse

def RandomErasing(img, probability = 0.5, sl = 0.02, sh = 0.2, r1 = 0.3, mean=[0.4914, 0.4822, 0.4465]):
    if random.uniform(0, 1) > probability:
        return img
    img_h=img.shape[0]
    img_w=img.shape[1]
    img_channel=img.shape[2]
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for attempt in range(100):
        area = img_h*img_w
        target_area = random.uniform(sl, sh) * area
        aspect_ratio = random.uniform(r1, 1/r1)
        h = int(round(math.sqrt(target_area * aspect_ratio)))
        w = int(round(math.sqrt(target_area / aspect_ratio)))
        if w < img_w and h < img_h:
            x1 = random.randint(0, img_h - h)
            y1 = random.randint(0, img_w - w)
            if img_channel == 3:
                img[x1:x1+h, y1:y1+w, 0] = mean[0]*255
                img[x1:x1+h, y1:y1+w, 1] = mean[1]*255
                img[x1:x1+h, y1:y1+w, 2] = mean[2]*255
            else:
                img[x1:x1+h, y1:y1+w, 0] = mean[0]*255
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    return img

def full_augment(class_path, save_path, aug_num):
    for sample in os.listdir(class_path):
        image = cv2.imread(f'{class_path}/{sample}')
        transform = A.Compose([
        A.Rotate(limit=20,p=1),
        A.OneOf([
            #A.RGBShift(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ],p=1),
        A.Flip(p=0.3),
        A.RandomResizedCrop(image.shape[0],image.shape[1],scale=(0.75, 1.0), ratio=(0.75, 1.3333333333333333),p=0.3),
        ])
        for temp in range(aug_num):
            transformed = transform(image=image)
            transformed_image = transformed["image"]
            #save aug image

            save_name = f'aug{temp}_'+''.join(random.sample(string.ascii_letters + string.digits, 32))+'.jpg'
            cv2.imwrite(os.path.join(f'{save_path}', save_name),transformed_image)
    print(f"augment class {aug_num} times complete")
                
def bagging_augment(class_path, save_path, ratio):
    sample_list = os.listdir(class_path)
    sample_select_num = int(len(sample_list)*ratio)
    print(f"bagging sample numble :{sample_select_num}")
    
    sample_selected = numpy.random.choice(sample_list, size=sample_select_num, replace=False)
    for sample in sample_selected:
        image = cv2.imread(f'{class_path}/{sample}')
        #augment pipline
        transform = A.Compose([
        A.Rotate(limit=20,p=1),
        A.OneOf([
            #A.RGBShift(p=0.3),
            A.RandomBrightnessContrast(p=0.3),
        ],p=1),
        A.Flip(p=0.3),
        A.RandomResizedCrop(image.shape[0],image.shape[1],scale=(0.75, 1.0), ratio=(0.75, 1.3333333333333333),p=0.3),
        ])
        
        transformed = transform(image=image)
        transformed_image = transformed["image"]
        #save aug image
        save_name = f'bagging_aug_'+''.join(random.sample(string.ascii_letters + string.digits, 32))+'.jpg'
        cv2.imwrite(os.path.join(f'{save_path}', save_name),transformed_image)
        
def copy_ori_dataset(args):
    path = f'{args.datapath}/split'
    size = len(os.listdir(f'{path}/train/'))
    index = 1
    for i in os.listdir(f'{path}/train/'):
        print(f'copy ori dataset  {index}/{size}')
        index+=1
        for j in os.listdir(f'{path}/train/{i}'):
            save_path = f'{path}/{args.aug_folder_name}/{i}'
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            shutil.copyfile(f'{path}/train/{i}/{j}',f'{save_path}/{j}')
    return True
    
def augment_inbalance(args):
    path = f'{args.datapath}/split/'
    size = len(os.listdir(f'{path}/train/'))
    index = 1
    sample_sum = 0
    for i in os.listdir(f'{path}/train/'):
        sample_sum += len(os.listdir(f'{path}/train/{i}'))
        
    print("ori dataset copy start")
    copy_ori_dataset(args)
    print("ori dataset copy complete")
    
    aug_per_class = sample_sum*args.augnum/size
    for i in os.listdir(f'{path}/train/'):
        print(f'process  {index}/{size}')
        index+=1
        
        class_size=len(os.listdir(f'{path}/train/{i}'))
        aug_ratio = aug_per_class/class_size
        print(f'augment class:{i}  ------- ratio:{aug_ratio}')
        if(aug_ratio<1):
            #bagging
            bagging_augment(f'{path}/train/{i}', f'{path}/{args.aug_folder_name}/{i}', aug_ratio)
        else:
            full_augment(f'{path}/train/{i}', f'{path}/{args.aug_folder_name}/{i}', int(aug_ratio))

            bagging_augment(f'{path}/train/{i}', f'{path}/{args.aug_folder_name}/{i}', aug_ratio-int(aug_ratio))
        print(f'augment class:{i}  ------- complete')
                
            
    
def augment_dataset(args):
    path = f'{args.datapath}/split/'
    size = len(os.listdir(f'{path}/train/'))
    index = 1
    # copy the ori dataset,anyway use this function first to create saved folder
    print("ori dataset copy start")
    copy_ori_dataset(args)
    print("ori dataset copy complete")
    
    for i in os.listdir(f'{path}/train/'):
        print(f'process  {index}/{size}')
        index+=1
        full_augment(f'{path}/train/{i}', f'{path}/{args.aug_folder_name}/{i}', args.augnum)
   
def main(args):
    if args.balanced_aug:
        print("augment dataset in balance mode")
        augment_inbalance(args)
    else:
        print("augment dataset in multiple mode")
        augment_dataset(args)
#         for data in os.listdir(args.datapath):
#             print(f'process-----------> {data}')
#             if data in ('sih-road-dataset', 'Emotion-Detection', 'yoga-pose', 'Flowers-Recognition', 'auto_datset_file_list.txt', 'MAMe-dataset', 'Boat-types-recognition', 'MIT-Indoor-Scenes', 'Chinese-MNIST', 'CUB_200_2011', 'food-101', 'fgvc-aircraft-2013b', 'Bald_Classification', 'cifar100', 'MIT-Indoor-Scenes.zip', 'casting_data', 'planes', 'A-Large-Scale-Fish-Dataset', 'sample_submission.csv', 'leaf-classification', 'the-nature-conservancy-fisheries-monitoring', 'cassava-leaf-diease', 'garbage_classification', 'CINIC-10', 'dogs-vs-cats-redux-kernels-edition'):
#                 continue
#             if os.path.isdir(f'{args.datapath}/{data}/'):
#                 augment_dataset(f'{args.datapath}/{data}/')
        
            
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datapath", help="train path of dataset ", type=str, default="")
    parser.add_argument("--aug_folder_name", help="augment data saved folder name ", type=str, default="train_dataaug")
    parser.add_argument("--augnum", help="num of image aug times ", type=int, default=2)
    parser.add_argument('--balanced_aug', action='store_true', default=False,help='augment the dataset in balance mode')
    args = parser.parse_args()
    main(args)
    
