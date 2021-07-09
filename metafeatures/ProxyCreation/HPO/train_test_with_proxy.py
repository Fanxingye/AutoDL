import time
import torch.nn as nn
import torch.utils
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.datasets as dset
import torch.backends.cudnn as cudnn
from torchvision import models
from torch.autograd import Variable

from split_dataset import split_train_test
from generate_entropy_file import write_loss_file, Net

def train(train_set, val_set, inputs, class_nums, args, reporter):
    epoch_num = args.epochs
    learning_rate = args.lr

    if (inputs.resnet18):
        net = models.resnet18(pretrained=True)
        net.fc=nn.Linear(inputs.fc_layers, class_nums)
    else:
        net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
    # Training stage
    for epoch in range(epoch_num):  # loop over the dataset multiple times
        running_loss = 0.0
        batch_num = 0.0
        epoch_start = time.time()

        for i, data in enumerate(train_set, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
            batch_num += 1
        print("epoch %d: loss: %f" % (epoch, (running_loss / batch_num)))
        print("         time %d sec. (expected finished time: after %.1f min.)" %(time.time() - epoch_start, (epoch_num-1-epoch)*(time.time()-epoch_start)/60))

        # Val stage
        correct = 0
        total = 0
        with torch.no_grad():
            for data in val_set:
                images, labels = data
                # calculate outputs by running images through the network
                outputs = net(images)
                # the class with the highest energy is what we choose as prediction
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        #reporter(epoch=epoch+1, accuracy=correct / total)
        print('         accuracy of the network on the validation set: %d %%' % (
        100 * correct / total))

    print('Finished Training')
    return net


def test(model, test_set):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_set:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the tester set: %d %%' % (
    100 * correct / total))
    return correct / total