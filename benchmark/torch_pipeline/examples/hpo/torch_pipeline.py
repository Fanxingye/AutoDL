import os
import time
import copy
import argparse
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
import autogluon.core as ag
from tqdm.auto import tqdm
from utils import from_argparse


logger = logging.getLogger(__name__)
def parse_args():
    parser = argparse.ArgumentParser(description='Model-based Asynchronous HPO')
    parser.add_argument('--debug', action='store_true',
                        help='debug if needed')
    parser.add_argument('--epochs', type=int, default=9,
                        help='number of epochs')
    parser.add_argument('--scheduler', type=str, default='fifo',
                        choices=['fifo', 'hyperband_stopping', 'hyperband_promotion'],
                        help='Scheduler name (default: fifo)')
    parser.add_argument('--random_seed', type=int, default=31415927,
                        help='random seed')
    # Note: 'model' == 'bayesopt' (legacy)
    parser.add_argument('--searcher', type=str, default='random',
                        choices=['random', 'model', 'bayesopt'],
                        help='searcher name (default: random)')
    # Arguments for FIFOScheduler
    parser.add_argument('--num_trials', type=int,
                        help='number of trial tasks')
    parser.add_argument('--scheduler_timeout', type=float, default=120,
                        help='maximum time until trials are started')
    # Arguments for HyperbandScheduler
    parser.add_argument('--brackets', type=int,
                        help='number of brackets')
    parser.add_argument('--reduction_factor', type=int,
                        help='Reduction factor for successive halving')
    parser.add_argument('--grace_period', type=int,
                        help='minimum number of epochs to run with'
                             'hyperband_* scheduler')
    parser.add_argument('--use_single_rung_system', action='store_true',
                        help='Use single rung level system for all brackets')
    args = parser.parse_args()
    return args


data_dir = '/media/robin/DATA/datatsets/image_data/hymenoptera/images'

def train_loop(args, reporter):
    learning_rate = args.learning_rate
    momentum = args.momentum
    num_epochs = args.epochs
    batch_size = args.batch_size

    device ='cuda' if torch.cuda.is_available() else 'cpu'
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                            data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=4, pin_memory=True)
                for x in ['train', 'val']}
    
    # Model
    net = models.resnet18(pretrained=True)
    num_ftrs = net.fc.in_features
    # Here the size of each output sample is set to 2.
    # Alternatively, it can be generalized to nn.Linear(num_ftrs, len(class_names)).
    net.fc = nn.Linear(num_ftrs, 2)

    net = net.to(device)
    if device == 'cuda':
        net = nn.DataParallel(net)
        
    criterion = nn.CrossEntropyLoss()
    # Observe that all parameters are being optimized
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=momentum)

    # Training
    def train(epoch):
        net.train()
        train_loss, correct, total = 0, 0, 0
        for batch_idx, (inputs, targets) in enumerate(dataloaders['train']):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    def test(epoch):
        net.eval()
        test_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(dataloaders['val']):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        acc = 100.*correct/total
        # 'epoch' reports the number of epochs done
        reporter(epoch=epoch+1, accuracy=acc)

    for epoch in tqdm(range(0, num_epochs)):
        train(epoch)
        test(epoch)


@ag.args(
    learning_rate=ag.space.Real(lower=1e-6, upper=1, log=True),
    momentum=ag.space.Real(lower=0.88, upper=0.9),
    batch_size=ag.space.Int(lower=8, upper=16),
    epochs=10,
)
def train_finetune(args, reporter):
    return train_loop(args, reporter)


if __name__ == '__main__':
    OPENML_TASK_ID = 6                # describes the problem we will tackle
    RATIO_TRAIN_VALID = 0.33          # split of the training data used for validation
    RESOURCE_ATTR_NAME = 'epoch'      # how do we measure resources   (will become clearer further)
    REWARD_ATTR_NAME = 'accuracy'    # how do we measure performance (will become clearer further)

    args = parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    # Build scheduler and searcher
    # scheduler_cls = ag.scheduler.FIFOScheduler if args.scheduler == 'fifo' \
    #     else ag.scheduler.HyperbandScheduler
    myscheduler = ag.scheduler.FIFOScheduler(train_finetune,
                                            resource={'num_cpus': 8, 'num_gpus': 1},
                                            checkpoint='checkpoint',
                                            num_trials=2,
                                            time_attr='epoch',
                                            reward_attr="accuracy")

    # Run experiment
    myscheduler.run()
    myscheduler.join_jobs()
    myscheduler.get_training_curves(plot=True, use_legend=False)
    print('The Best Configuration and Accuracy are: {}, {}'.format(myscheduler.get_best_config(),
                                                                myscheduler.get_best_reward()))
    logger.info("Finished joining all tasks!")