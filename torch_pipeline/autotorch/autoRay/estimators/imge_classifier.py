import argparse
import logging
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.parallel import DistributedDataParallel
import ray
from ray import tune
from ray.tune.integration.torch import (DistributedTrainableCreator,
                                        distributed_checkpoint_dir)


class ImageClassifier():
    def __init__(self,
                 config,
                 logger=None,
                 reporter=None,
                 net=None,
                 optimizer=None,
                 problem_type=None):

    def fit():
        return NotImplemented
    
    def train_loop(config, checkpoint_dir=False):
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

        for epoch in range(10):
            train(model, optimizer, train_loader, device)
            acc = test(model, test_loader, device)

            if epoch % 3 == 0:
                with distributed_checkpoint_dir(step=epoch) as checkpoint_dir:
                    path = os.path.join(checkpoint_dir, "checkpoint")
                    torch.save((model.state_dict(), optimizer.state_dict()), path)
            tune.report(mean_accuracy=acc)


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
