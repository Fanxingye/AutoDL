"""Utils for auto classification estimator"""
# pylint: disable=bad-whitespace,missing-function-docstring
import os
import math
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms
from .dataset import TorchImageClassificationDataset


def get_dataset(data_dir, input_size, crop_ratio, train_dataset=None, val_dataset=None):
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
    lighting_param = 0.1
    crop_ratio = crop_ratio if crop_ratio > 0 else 0.875
    resize = int(math.ceil(input_size / crop_ratio))

    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(input_size),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=jitter_param, contrast=jitter_param, saturation=jitter_param),
        transforms.ToTensor(),
        normalize
    ])
    transform_test = transforms.Compose([
        transforms.Resize(resize),
        transforms.CenterCrop(input_size),
        transforms.ToTensor(),
        normalize
    ])

    if train_dataset is None:
        train_dataset = datasets.ImageFolder(os.path.join(data_dir, 'train'), transform_train)
    elif isinstance(train_dataset, TorchImageClassificationDataset):
        train_dataset = train_dataset.to_pytorch(transform_train)
    else:
        train_dataset = None
    if not val_dataset:
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform_test)
    elif isinstance(val_dataset, TorchImageClassificationDataset):
        val_dataset = train_dataset.to_pytorch(transform_test)
    else:
        val_dataset = None

    return train_dataset, val_dataset


def get_data_loader(batch_size, num_workers):
    train_data = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size,
                                            shuffle=True,
                                            num_workers=num_workers,
                                            pin_memory=True)
    val_data = torch.utils.data.DataLoader(val_dataset,
                                           batch_size=batch_size,
                                           shuffle=False,
                                           num_workers=num_workers,
                                           pin_memory=True)
    return train_data, val_data
