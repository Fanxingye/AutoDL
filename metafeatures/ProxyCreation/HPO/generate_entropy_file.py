import os
import sys
import time
import glob
import numpy as np
import torch
import logging
import argparse
import torch.nn as nn
import torch.utils
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torchvision import models

batch_size = 32


def write_loss_file(train_data, args):
    # Loading data
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=False, num_workers=2)

    # Build naive network
    criterion = nn.CrossEntropyLoss(reduce=False)
    class_nums = len(train_data.classes)

    if (args.resnet50):
        net = models.resnet50(pretrained=True)
        net.fc=nn.Linear(args.fc_layers, class_nums)
        filename = 'optimal_config_' + args.dataset + '_proxy_resnet50_entropy_file.txt'
    elif (args.resnet18):
        net = models.resnet18(pretrained=True)
        net.fc=nn.Linear(args.fc_layers, class_nums)
        filename = 'optimal_config_' + args.dataset + '_proxy_resnet18_entropy_file.txt'
    else:
        net = Net()
        filename = 'optimal_config_' + args.dataset + '_proxy_naiveNet_entropy_file.txt'


    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # Three colums to store in a entropy file
    index = []
    entropy = []
    label = []

    # Get loss for each sample:
    # 1st epoch: train the naive network; 2nd epoch: evaluate loss for each epoch
    for epoch in range(2):
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            #print(i)
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.sum().backward()
            optimizer.step()
            
            # Store three columns to be wroten
            if (epoch == 1):
                for idx in range(len(labels)):
                    index.append(batch_size * int(i) + int(idx))
                    entropy.append((float)(loss[idx].item()) + 0.00002)                    
                    label.append((int)(labels[idx]))
            
            # print statistics
            running_loss += loss.sum().item() / batch_size
            if i % 200 == 199:    # print every 200 mini-batches
                #print((float)(loss[0].item()), (int)(labels[0]))
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 200))
                running_loss = 0.0
    print('Finished Training')

    with open(os.path.join('../entropy_list', filename), 'w') as f:
        for idx in index:
            f.write('%d %f %d\n'%(idx, entropy[idx], label[idx]))
    print('Finished Writinging')
    return '../entropy_list/' + filename

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(1296, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
