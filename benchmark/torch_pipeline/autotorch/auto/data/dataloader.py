"""Utils for auto classification estimator"""
# pylint: disable=bad-whitespace,missing-function-docstring
import os
import math
import torch
from torchvision import datasets, transforms
from .dataset import TorchImageClassificationDataset


def get_data_loader(data_dir, batch_size, num_workers, input_size, crop_ratio, data_augment, train_dataset=None, val_dataset=None):
    """AutoPytorch ImageClassification data loaders
    Parameters:
    -----------
    data_dir: 
        data_dir
    batch_size : 
        batch_szie
    num_workers: 
        4
    input_size:
         224
    crop_ratio  :
        0.875
    data_augment : 
        None
    train_dataset : 
        TorchImageClassificationDataset
    val_dataset   : 
        TorchImageClassificationDataset
    """
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    jitter_param = 0.4
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
    else:
        assert isinstance(train_dataset, TorchImageClassificationDataset), "DataSet Type Error"
        train_dataset = train_dataset.to_pytorch(transform_train)

    if val_dataset is None:
        val_dataset = datasets.ImageFolder(os.path.join(data_dir, 'val'), transform_test)
    else:
        assert isinstance(val_dataset, TorchImageClassificationDataset), "DataSet Type Error"
        val_dataset = val_dataset.to_pytorch(transform_test)

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
