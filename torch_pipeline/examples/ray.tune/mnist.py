# Original Code here:
# https://github.com/pytorch/examples/blob/master/mnist/main.py
import argparse
import logging
import os
import torch
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import ray
from ray import tune
from ray.tune.examples.mnist_pytorch import get_data_loaders
from ray.tune.integration.torch import (DistributedTrainableCreator,
                                        distributed_checkpoint_dir)

logger = logging.getLogger(__name__)

# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


def train(model, optimizer, train_loader, device=None):
    device = device or torch.device("cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device=None):
    device = device or torch.device("cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def train_mnist(config, checkpoint_dir=False):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    train_loader, test_loader = get_data_loaders()
    model = ConvNet().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.1)

    if checkpoint_dir:
        with open(os.path.join(checkpoint_dir, "checkpoint")) as f:
            model_state, optimizer_state = torch.load(f)

        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    model = DistributedDataParallel(model)

    for epoch in range(400):
        train(model, optimizer, train_loader, device)
        acc = test(model, test_loader, device)

        if epoch % 3 == 0:
            with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)
        tune.report(mean_accuracy=acc)


def run_ddp_tune(num_workers, num_gpus_per_worker, workers_per_node=None):
    trainable_cls = DistributedTrainableCreator(
        train_mnist,
        num_workers=num_workers,
        num_gpus_per_worker=num_gpus_per_worker,
        num_workers_per_host=workers_per_node)

    analysis = tune.run(trainable_cls,
                        num_samples=4,
                        stop={"training_iteration": 10},
                        metric="mean_accuracy",
                        mode="max")

    print("Best hyperparameters found were: ", analysis.best_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-workers",
                        "-n",
                        type=int,
                        default=2,
                        help="Sets number of workers for training.")
    parser.add_argument("--num-gpus-per-worker",
                        type=int,
                        default=0,
                        help="Sets number of gpus each worker uses.")
    parser.add_argument("--cluster",
                        action="store_true",
                        default=False,
                        help="enables multi-node tuning")
    parser.add_argument(
        "--workers-per-node",
        type=int,
        help="Forces workers to be colocated on machines if set.")
    parser.add_argument("--server-address",
                        type=str,
                        default=None,
                        required=False,
                        help="The address of server to connect to if using "
                        "Ray Client.")

    args = parser.parse_args()

    if args.server_address is not None:
        ray.util.connect(args.server_address)
    else:
        if args.cluster:
            options = dict(address="auto")
        else:
            options = dict(num_cpus=8)
        ray.init(**options)

    run_ddp_tune(num_workers=args.num_workers,
                 num_gpus_per_worker=args.num_gpus_per_worker,
                 workers_per_node=args.workers_per_node)
