import os
import time
import torch
import logging
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import torch.distributed as dist
from autotorch.models.network import init_network, get_input_size
from autotorch.auto.data.dataloader import get_pytorch_train_loader, get_pytorch_val_loader
from autotorch.utils.metrics import AverageMeter, accuracy, ProgressMeter
from autotorch.utils.model import reduce_tensor
from autotorch.auto.estimators.base_estimator import set_default
from autotorch.proxydata.default import ImageClassificationCfg
from autotorch.proxydata.sampler import read_entropy_file, get_proxy_data_random, get_proxy_data_log_entropy_histogram


@set_default(ImageClassificationCfg())
class ProxyModel():
    def __init__(self, config=None, logger=None, name=None):
        self._logger = logger if logger is not None else logging.getLogger(
            name)
        self._logger.setLevel(logging.INFO)
        self._cfg = self._default_cfg

    def fit(self, train_data, val_data=None):

        self.distributed = False
        if "WORLD_SIZE" in os.environ:
            self.distributed = int(os.environ["WORLD_SIZE"]) > 1
            self.local_rank = int(os.environ["LOCAL_RANK"])
        else:
            self.local_rank = 0

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.distributed = False
        world_size = 1
        if self.distributed:
            self.gpu = self.local_rank % torch.cuda.device_count()

            if not torch.distributed.is_initialized():
                torch.cuda.set_device(self.gpu)
                dist.init_process_group(backend="nccl", init_method="env://")
                world_size = torch.distributed.get_world_size()

        # init network
        classes = train_data.classes
        num_class = len(classes)

        model_name = self._cfg.img_cls.model_name.lower()
        self.input_size = get_input_size(model_name)
        use_pretrained = self._cfg.img_cls.use_pretrained
        self.net = init_network(model_name=model_name,
                                num_class=num_class,
                                pretrained=use_pretrained)

        if self.distributed:
            # For multiprocessing distributed, DistributedDataParallel constructor
            # should always set the single device scope, otherwise,
            # DistributedDataParallel will use all available devices.
            if self.gpu is not None:
                torch.cuda.set_device(self.gpu)
                self.net.cuda(self.device)
                # When using a single GPU per process and per
                # DistributedDataParallel, we need to divide the batch size
                # ourselves based on the total number of GPUs we have
                self.net = torch.nn.parallel.DistributedDataParallel(
                    self.net, device_ids=[self.gpu], output_device=self.gpu)
            else:
                self.net.cuda(self.device)
                # DistributedDataParallel will divide and allocate batch_size to all
                # available GPUs if device_ids are not set
                self.net = torch.nn.parallel.DistributedDataParallel(
                    self.net, output_device=0)
        else:
            self.net.cuda(self.device)

        # init loss function
        loss = nn.CrossEntropyLoss()
        self.criterion = loss.to(self.device)
        # init optimizer
        self.optimizer = optim.SGD(params=self.net.parameters(),
                                   lr=self._cfg.train.base_lr,
                                   momentum=self._cfg.train.momentum,
                                   weight_decay=self._cfg.train.weight_decay,
                                   nesterov=self._cfg.train.nesterov)

        train_loader = get_pytorch_train_loader(
            data_dir=self._cfg.train.data_dir,
            batch_size=self._cfg.train.batch_size,
            num_workers=self._cfg.train.num_workers,
            input_size=self.input_size,
            crop_ratio=self._cfg.train.crop_ratio,
            data_augment=self._cfg.train.data_augment,
            train_dataset=train_data)

        val_loader = get_pytorch_val_loader(
            data_dir=self._cfg.train.data_dir,
            batch_size=self._cfg.valid.batch_size,
            num_workers=self._cfg.train.num_workers,
            input_size=self.input_size,
            crop_ratio=self._cfg.train.crop_ratio,
            val_dataset=val_data)

        for epoch in range(0, self._cfg.train.epochs):
            # train for one epoch
            self.train(train_loader, self.net, self.criterion, self.optimizer,
                       epoch, num_class)
            # evaluate on validation set
            acc1 = self.validate(val_loader, self.net, self.criterion,
                                 num_class)

    def train(self, train_loader, model, criterion, optimizer, epoch,
              num_class):
        batch_time = AverageMeter('Time', ':6.3f')
        data_time = AverageMeter('Data', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(train_loader),
                                 [batch_time, data_time, losses, top1, top5],
                                 prefix="Epoch: [{}]".format(epoch))

        # switch to train mode
        model.train()
        end = time.time()
        for i, (images, target) in enumerate(train_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            images = images.to(self.device)
            target = target.to(self.device)

            # compute output
            output = model(images)
            loss = criterion(output, target)
            if self.distributed:
                reduced_loss = reduce_tensor(loss.data)
                losses.update(reduced_loss.item(), images.size(0))
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, min(5, num_class)))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))
            # compute gradient and do SGD step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if self.local_rank == 0:
                if i % self._cfg.train.log_interval == 0:
                    progress.display(i)

    def validate(self, val_loader, model, criterion, num_class):
        batch_time = AverageMeter('Time', ':6.3f')
        losses = AverageMeter('Loss', ':.4e')
        top1 = AverageMeter('Acc@1', ':6.2f')
        top5 = AverageMeter('Acc@5', ':6.2f')
        progress = ProgressMeter(len(val_loader),
                                 [batch_time, losses, top1, top5],
                                 prefix='Test: ')

        # switch to evaluate mode
        model.eval()

        with torch.no_grad():
            end = time.time()
            for i, (images, target) in enumerate(val_loader):
                images = images.to(self.device)
                target = target.to(self.device)
                # compute output
                output = model(images)
                loss = criterion(output, target)

                # measure accuracy and record loss
                acc1, acc5 = accuracy(output,
                                      target,
                                      topk=(1, min(5, num_class)))
                losses.update(loss.item(), images.size(0))
                top1.update(acc1.item(), images.size(0))
                top5.update(acc5.item(), images.size(0))

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()

                if i % self._cfg.valid.log_interval == 0:
                    progress.display(i)

            # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(
                top1=top1, top5=top5))
        self.net = model
        return top1.avg

    def generate_proxy_data(self,
                            val_data,
                            sampling_portion=0.2,
                            sampler_type='histogram',
                            output_dir='',
                            entropy_file_name='entropy_list.txt',
                            logging=None):
        def get_entropy(data):
            entropy = F.softmax(data, dim=1) * F.log_softmax(data + 1e-6,
                                                             dim=1)
            entropy = -1.0 * entropy.sum(dim=1)
            return entropy

        batch_size = self._cfg.test.batch_size
        data_loader = get_pytorch_val_loader(
            data_dir=self._cfg.train.data_dir,
            batch_size=batch_size,
            num_workers=self._cfg.test.num_workers,
            input_size=self.input_size,
            crop_ratio=self._cfg.train.crop_ratio,
            val_dataset=val_data)

        index_list = []
        entropy_list = []  # result
        label_list = []
        for i, (input, target) in enumerate(data_loader):
            input = input.to(self.device)
            target = target.to(self.device)
            output = self.net(input)  # model must be declared
            ent = get_entropy(output.data)  # entropy extraction

            # generate entropy file
            batch_list = list(range(batch_size * i, batch_size * (i + 1)))
            index_list.extend(batch_list)
            entropy_list.extend(ent.tolist())
            label_list.extend(target.data.tolist())

        # write the entropy file
        entropy_path = os.path.join(output_dir, entropy_file_name)
        with open(entropy_path, 'w') as f:
            for idx in index_list:
                f.write('%d %f %d\n' %
                        (idx, entropy_list[idx], label_list[idx]))

        # generate proxy dataset
        index, entropy, label = read_entropy_file(entropy_path)

        if sampler_type == 'random':
            indices = get_proxy_data_random(entropy, sampling_portion, logging)
        elif sampler_type == 'histogram':
            indices = get_proxy_data_log_entropy_histogram(entropy,
                                                           sampling_portion,
                                                           histogram_type=1,
                                                           dataset='cifar10',
                                                           logging=logging)
        else:
            raise NotImplementedError

        val_data = val_data.iloc[indices, :]
        return val_data


if __name__ == '__main__':
    from autotorch.auto import ImagePredictor
    train_dataset, _, test_dataset = ImagePredictor.Dataset.from_folders(
        'https://autogluon.s3.amazonaws.com/datasets/shopee-iet.zip')

    proxmodel = ProxyModel()
    proxmodel.fit(train_dataset, test_dataset)
    proxy_data = proxmodel.generate_proxy_data(train_dataset)