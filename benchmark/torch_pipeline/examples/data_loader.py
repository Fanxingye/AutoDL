import os
import torch
import torch.utils.data
from torchvision import datasets, models, transforms

def load_data(data_path, rescaled_size, batch_size, ngpus_per_node):
    ngpus_per_node = max(ngpus_per_node, 1)
    test_batch_size = int(batch_size / ngpus_per_node)
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(rescaled_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(rescaled_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(rescaled_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_path, x),
                                            data_transforms[x])
                    for x in ['train', 'val', 'test']}
    dataloaders = {'train': torch.utils.data.DataLoader(image_datasets['train'], batch_size=batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True),
                   'val': torch.utils.data.DataLoader(image_datasets['val'], batch_size=batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True),
                   'test': torch.utils.data.DataLoader(image_datasets['test'], batch_size=test_batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)
                    }
    class_names = image_datasets['train'].classes
    num_class = len(class_names)
    return dataloaders, num_class
