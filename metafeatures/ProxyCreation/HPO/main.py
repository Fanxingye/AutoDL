import torch
import torchvision.datasets as dset
import torchvision.transforms as transforms
import numpy as np
import argparse
import autogluon.core as ag

import torchvision
from split_dataset import split_train_test
from generate_entropy_file import write_loss_file, Net
from create_proxy import subsampling
from train_test_with_proxy import train, test

# TO　DO: transforms.Resize() should be auto-adjusting based on input network.
# default resnet18 here
transform = transforms.Compose(
    [transforms.Resize([56, 56]),transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

parser = argparse.ArgumentParser("p")
parser.add_argument('--proxy', action='store_true', default=False, help='use histogram-based sampling')
parser.add_argument('--original', action='store_true', default=False, help='use original entropy file')
parser.add_argument('--resnet50', action='store_true', default=False, help='use resnet50')
parser.add_argument('--resnet18', action='store_true', default=False, help='use resnet18')
parser.add_argument('--sampling_portion', type=float, default=0.2, help='proxy dataset relative size to the target dataset')
parser.add_argument('--train_portion', type=float, default=0.8, help='portion of training data')
parser.add_argument('--fc_layers', type=int, default=512, help='resnet50 fc layers')
parser.add_argument('--dataset', type=str, default='cifar10', help='target data [cifar10, emotion_detection, leaf_classification, dog-breed-identification]')

inputs = parser.parse_args()

def main(args, reporter):

    #Data Preprocessing
    if (inputs.dataset == 'cifar10'):
        train_path = "/data/AutoML_compete/cifar10/train"
        test_path = "/data/AutoML_compete/cifar10/test"
    elif (inputs.dataset == 'emotion_detection'):
        train_path = "/data/AutoML_compete/Emotion-Detection/train"
        test_path = "/data/AutoML_compete/Emotion-Detection/split/test"
    elif (inputs.dataset == 'leaf_classification'):
        train_path = "/data/AutoML_compete/leaf-classification/train"
        test_path = "/data/AutoML_compete/leaf-classification/split/test"
    elif (inputs.dataset == 'dog_breed_identification'):
        train_path = "/data/AutoML_compete/dog-breed-identification/train"
        test_path = "/data/AutoML_compete/dog-breed-identification/split/test"
    

    data = torchvision.datasets.ImageFolder(root=train_path, transform=transform)
    test_data = torchvision.datasets.ImageFolder(root=test_path, transform=transform)
    class_nums = len(data.classes)

    # Writing entropy loss for every sample
    entropy_file = write_loss_file(data, inputs)
    print("finished Writing entropy file\n")

    if inputs.proxy:
        if (inputs.original):
            # using author provided entropy file
            indices = subsampling('../entropy_list/cifar10_resnet20_index_entropy_class.txt', inputs.sampling_portion)
        else:
            # using previsouly written entropy file to perform subsampling
            # getting a list of sample index to form proxy dataset
            indices = subsampling(entropy_file, inputs.sampling_portion)

        num_train = num_proxy_data = len(indices)
        split = int(np.floor(inputs.train_portion * num_proxy_data))
        train_set = torch.utils.data.DataLoader(
        data, batch_size=args.bs,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[:]),
        num_workers=2)

        val_set = torch.utils.data.DataLoader(
        data, batch_size=args.bs,
        sampler=torch.utils.data.sampler.SubsetRandomSampler(indices[split:num_train]),
        num_workers=2)

        test_set = torch.utils.data.DataLoader(
        test_data, batch_size=args.bs,
        num_workers=2)
        print("in proxy set")
    else:
    # full dataset
        train_set, val_set = split_train_test(data, inputs.train_portion)
        train_set = torch.utils.data.DataLoader(train_set, batch_size=args.bs, shuffle=False, num_workers=2)
        val_set = torch.utils.data.DataLoader(val_set, batch_size=args.bs, shuffle=False, num_workers=2)
        test_set = torch.utils.data.DataLoader(test_data, batch_size=args.bs, shuffle=False, num_workers=2)
        print("in full set")
    
    print("train size: %d, val size: %d, test size: %d\n"
    %(len(train_set) * args.bs, len(val_set) * args.bs, len(test_set) * args.bs))
    print("learning rate: %f, batch size: %d\n" %(args.lr, args.bs))
    model = train(train_set, val_set, inputs, class_nums, args, reporter)
    test_acc = test(model, test_set)
    reporter(accuracy=test_acc)

@ag.args(
    lr = ag.space.Real(0.001, 0.1, log=True),
    bs = 64,
    epochs= 50,
)

def ag_train(args, reporter):
    return main(args, reporter)

myscheduler = ag.scheduler.FIFOScheduler(
    ag_train,
    resource={'num_gpus': 2},
    num_trials=50,
    reward_attr='accuracy')
print(myscheduler)

myscheduler.run()
myscheduler.join_jobs()
print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                               myscheduler.get_best_reward()))

if (inputs.proxy):
    f = open('optimal_config_proxy.txt',"w")
else:
    f = open('optimal_config_full.txt',"w")
f.write(str(myscheduler.get_best_config()))
f.close()