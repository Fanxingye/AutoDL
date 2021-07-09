import torch
import torchvision.datasets as dset

def split_train_test(data, train_portion):
    train_size = (int)(len(data) * train_portion)
    val_size = (int)(len(data) - train_size)
    train_set, test_set = torch.utils.data.random_split(data, [train_size, val_size])
    return train_set, test_set
