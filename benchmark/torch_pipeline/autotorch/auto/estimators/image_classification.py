"""Classification Estimator"""
# pylint: disable=unused-variable,bad-whitespace,missing-function-docstring,logging-format-interpolation,arguments-differ,logging-not-lazy
import time
import os
import math
import copy

from PIL import Image
import pandas as pd
import numpy as np

import torch
from torch import optim
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data
from torch.cuda.amp import autocast
from torch.autograd import Variable
import torch.utils.data.distributed
from mmcv.runner import get_dist_info, init_dist
from autotorch.data import *
from autotorch.data.mixup import NLLMultiLabelSmooth, MixUpWrapper
from autotorch.data.smoothing import LabelSmoothing
from autotorch.models.model_zoo import get_model_list
from autotorch.models.network import init_network, get_input_size
from autotorch.optim.optimizers import get_optimizer
from autotorch.scheduler.lr_scheduler import *
from autotorch.utils.model import resum_checkpoint
from autotorch.utils.model import reduce_tensor, save_checkpoint
from autotorch.utils.metrics import AverageMeter, accuracy

import autogluon.core as ag
from .base_estimator import BaseEstimator, set_default
from .default import ImageClassificationCfg
from ..data.dataset import TorchImageClassificationDataset
from ..data.dataloader import get_data_loader
from ..conf import _BEST_CHECKPOINT_FILE
from gluoncv.auto.estimators.utils import EarlyStopperOnPlateau



__all__ = ['ImageClassificationEstimator']


@set_default(ImageClassificationCfg())
class ImageClassificationEstimator(BaseEstimator):
    """Estimator implementation for Image Classification.

    Parameters
    ----------
    config : dict
        Config in nested dict.
    logger : logging.Logger
        Optional logger for this estimator, can be `None` when default setting is used.
    reporter : callable
        The reporter for metric checkpointing.
    net : torch.Module
        The custom network. If defined, the model name in config will be ignored so your
        custom network will be used for training rather than pulling it from model zoo.
    """
    Dataset = TorchImageClassificationDataset

    def __init__(self, config, logger=None, reporter=None, net=None, optimizer=None):
        super(ImageClassificationEstimator, self).__init__(config, logger=logger, reporter=reporter, name=None)
        self.last_train = None
        self.input_size = self._cfg.train.input_size

        if net is not None:
            assert isinstance(net, torch.nn.Module), f"given custom network {type(net)}, torch.nn.Module expected"
        self._custom_net = net

        if optimizer is not None:
            if isinstance(optimizer, str):
                pass
            else:
                assert isinstance(optimizer, torch.optim.Optimizer)
        self._optimizer = optimizer

    def _init_dist_envs(self):
        # set cudnn_benchmark
        if self._cfg.get('cudnn_benchmark', False):
            torch.backends.cudnn.benchmark = True

        if self._cfg.gpus is not None:
            self.gpu_ids = self._cfg.gpus
        else:
            self.gpu_ids = range(1) if self._cfg.gpus is None else range(self._cfg.gpus)

        # init distributed env first, since logger depends on the dist info.
        if self._cfg.launcher == 'none':
            self.distributed = False
        else:
            self.distributed = True
            init_dist(self._cfg.launcher, backend='nccl')
            # re-set gpu_ids with distributed training mode
            _, world_size = get_dist_info()
            self.gpu_ids = range(world_size)

        if self.distributed:
            torch.cuda.set_device(self.gpu_ids)
            self.net = self.net.cuda(self.gpu_ids)
            self.net = torch.nn.parallel.DistributedDataParallel(self.net, device_ids=[self.gpu_ids], output_device=self.gpu_ids)
        else:
            torch.cuda.set_device(self.gpu_ids)
            self.net = self.net.cuda(self.gpu_ids)

    def _fit(self, train_data, val_data, time_limit=math.inf):
        tic = time.time()
        self._best_acc = -float('inf')
        self.epoch = 0
        self._time_elapsed = 0
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}
        if not isinstance(train_data, pd.DataFrame):
            self.last_train = len(train_data)
        else:
            self.last_train = train_data
        self._init_dist_envs()
        self._init_trainer()
        self._time_elapsed += time.time() - tic
        return self._resume_fit(train_data, val_data, time_limit=time_limit)

    def _resume_fit(self, train_data, val_data, time_limit=math.inf):
        tic = time.time()
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}

        num_workers = self._cfg.train.num_workers
        train_loader, val_loader = get_data_loader(self._cfg.train.data_dir,
                                                    self.batch_size, num_workers,
                                                    self.input_size,
                                                    self._cfg.train.crop_ratio,
                                                    self._cfg.train.data_augment,
                                                    train_dataset=train_data,
                                                    val_dataset=val_data)
        self._time_elapsed += time.time() - tic
        return self._train_loop(train_loader, val_loader, time_limit=time_limit)

    def _train_loop(self, train_data, val_data, time_limit=math.inf):
        start_tic = time.time()

        self._logger.info('Start training from [Epoch %d]', max(self._cfg.train.start_epoch, self.epoch))
        early_stopper = EarlyStopperOnPlateau(
            patience=self._cfg.train.early_stop_patience,
            min_delta=self._cfg.train.early_stop_min_delta,
            baseline_value=self._cfg.train.early_stop_baseline,
            max_value=self._cfg.train.early_stop_max_value)

        self._time_elapsed += time.time() - start_tic

        for self.epoch in range(max(self._cfg.train.start_epoch, self.epoch), self._cfg.train.epochs):
            epoch = self.epoch

            if self._best_acc >= 1.0:
                self._logger.info('[Epoch {}] Early stopping as acc is reaching 1.0'.format(epoch))
                break

            should_stop, stop_message = early_stopper.get_early_stop_advice()
            if should_stop:
                self._logger.info('[Epoch {}] '.format(epoch) + stop_message)
                break

            tic = time.time()
            losses_m, top1_m, top5_m = self._train_epoch(train_loader=train_data,
                                                        model=self.net,
                                                        criterion=self.criterion,
                                                        optimizer=self.optimizer,
                                                        scaler=self.scaler,
                                                        lr_scheduler=self.lr_policy,
                                                        num_class=self.num_class,
                                                        epoch=epoch,
                                                        use_amp=self._cfg.train.amp,
                                                        batch_size_multiplier=1,
                                                        logger=self._logger,
                                                        log_interval=10)

            # post_tic = time.time()
            # # steps_per_epoch = len(train_loader)
            # # throughput = int(self.batch_size * steps_per_epoch /(time.time() - tic))

            # self._logger.info('[Epoch %d] training: %s=%f', epoch, train_metric_name, train_metric_score)
            # # self._logger.info('[Epoch %d] speed: %d samples/sec\ttime cost: %f', epoch, throughput, time.time()-tic)

            top1_val, top5_val = self._val_epoch(val_loader=val_data,
                                                model=self.net,
                                                criterion=self.criterion,
                                                num_class=self.num_class,
                                                use_amp=self._cfg.train.amp,
                                                logger=self._logger,
                                                log_name="Val-log",
                                                log_interval=10
                                                )
            early_stopper.update(top1_val)
            self._logger.info('[Epoch %d] validation: top1=%f top5=%f', epoch, top1_val, top5_val)
            if top1_val > self._best_acc:
                cp_name = os.path.join(self._logdir, _BEST_CHECKPOINT_FILE)
                self._logger.info('[Epoch %d] Current best top-1: %f vs previous %f, saved to %s',
                                    self.epoch, top1_val, self._best_acc, cp_name)

                self.save(cp_name)
                self._best_acc = top1_val
                if self._reporter:
                    self._reporter(epoch=epoch, acc_reward=top1_val)
            self._time_elapsed += time.time() - tic

        return {'train_acc': train_metric_score, 'valid_acc': self._best_acc,
                'time': self._time_elapsed, 'checkpoint': cp_name}

    def _train_step(self, model, criterion, optimizer, scaler, use_amp=False, batch_size_multiplier=1, top_k=1):
        def step_fn(input, target, optimizer_step=True):
            input_var = Variable(input)
            target_var = Variable(target)

            with autocast(enabled=use_amp):
                output = model(input_var)
                loss = criterion(output, target_var)
                loss /= batch_size_multiplier

                prec1, prec5 = accuracy(output, target, topk=(1, min(top_k, 5)))
                if torch.distributed.is_initialized():
                    reduced_loss = reduce_tensor(loss.data)
                    prec1 = reduce_tensor(prec1)
                    prec5 = reduce_tensor(prec5)
                else:
                    reduced_loss = loss.data

            scaler.scale(loss).backward()
            if optimizer_step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            torch.cuda.synchronize()

            return reduced_loss, prec1, prec5
        return step_fn

    def _val_step(self, model, criterion, use_amp=False, top_k=1):
        def step_fn(input, target):
            input_var = Variable(input)
            target_var = Variable(target)

            with torch.no_grad(), autocast(enabled=use_amp):
                output = model(input_var)
                loss = criterion(output, target_var)

                prec1, prec5 = accuracy(output.data, target, topk=(1, min(5, top_k)))

                if torch.distributed.is_initialized():
                    reduced_loss = reduce_tensor(loss.data)
                    prec1 = reduce_tensor(prec1)
                    prec5 = reduce_tensor(prec5)
                else:
                    reduced_loss = loss.data

            torch.cuda.synchronize()

            return reduced_loss, prec1, prec5

        return step_fn

    def _train_epoch(self,
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    lr_scheduler,
                    num_class,
                    epoch,
                    use_amp=False,
                    batch_size_multiplier=1,
                    logger=None,
                    log_interval=10):
        """Compute a single epoch of train or validation.

        Parameters
        ----------
        train_loader : torch Dataset or None
          The initialized dataset to loop over. If None, skip this step.

        model :  model
          Whether to set the module to train mode or not.

        criterion : str
          Prefix to use when saving to the history.

        optimizer : callable
          Function to call for each batch.

        scaler  : scaler


        **fit_params : dict
          Additional parameters passed to the ``step_fn``.
        """

        batch_time_m = AverageMeter('BatchTime', ':6.3f')
        data_time_m = AverageMeter('DataTime', ':6.3f')
        losses_m = AverageMeter('Loss', ':.4e')
        top1_m = AverageMeter('Acc@1', ':6.2f')
        top5_m = AverageMeter('Acc@5', ':6.2f')

        step = self._train_step(
            model,
            criterion,
            optimizer,
            scaler=scaler,
            use_amp=use_amp,
            batch_size_multiplier=batch_size_multiplier,
            top_k=self.num_class
        )

        model.train()
        optimizer.zero_grad()
        steps_per_epoch = len(train_loader)
        end = time.time()

        for i, (input, target) in enumerate(train_loader):
            input = input.cuda()
            target = target.cuda()

            bs = input.size(0)
            lr_scheduler(optimizer, i, epoch)
            data_time = time.time() - end

            optimizer_step = ((i + 1) % batch_size_multiplier) == 0
            loss, prec1, prec5 = step(input, target, optimizer_step=optimizer_step)

            it_time = time.time() - end

            batch_time_m.update(it_time)
            data_time_m.update(data_time)
            losses_m.update(loss.item(), bs)
            top1_m.update(prec1.item(), bs)
            top5_m.update(prec5.item(), bs)

            end = time.time()
            if ((i+1) % log_interval == 0) or (i == steps_per_epoch - 1):
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    learning_rate = optimizer.param_groups[0]["lr"]
                    log_name = 'Train-log'
                    logger.info(
                        "{0}: [epoch:{1:>2d}] [{2:>2d}/{3}] "
                        'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'BatchTime: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f}) '
                        'lr: {lr:>4.6f} '.format(
                            log_name, epoch+1, i, steps_per_epoch, data_time=data_time_m,
                            batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m, lr=learning_rate))

        return losses_m.avg, top1_m.avg, top5_m.avg

    def _val_epoch(self,
                    val_loader,
                    model,
                    criterion,
                    num_class,
                    use_amp=False,
                    logger=None,
                    logger_name='Val-log',
                    log_interval=10):

        batch_time_m = AverageMeter('Time', ':6.3f')
        data_time_m = AverageMeter('Data', ':6.3f')
        losses_m = AverageMeter('Loss', ':.4e')
        top1_m = AverageMeter('Acc@1', ':6.2f')
        top5_m = AverageMeter('Acc@5', ':6.2f')

        step = self._val_step(model, criterion, use_amp=use_amp, top_k=num_class)
        # switch to evaluate mode
        model.eval()
        steps_per_epoch = len(val_loader)
        end = time.time()
        data_iter = enumerate(val_loader)

        for i, (input, target) in data_iter:
            bs = input.size(0)
            data_time = time.time() - end
            loss, prec1, prec5 = step(input, target)
            it_time = time.time() - end
            end = time.time()

            batch_time_m.update(it_time)
            data_time_m.update(data_time)
            losses_m.update(loss.item(),  bs)
            top1_m.update(prec1.item(), bs)
            top5_m.update(prec5.item(), bs)

            if ((i+1) % log_interval == 0) or (i == steps_per_epoch - 1):
                if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
                    logger.info(
                        '{0}: [{1:>2d}/{2}] '
                        'DataTime: {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Time: {batch_time.val:.3f} ({batch_time.avg:.3f}) '
                        'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f}) '
                        'Acc@1: {top1.val:>7.4f} ({top1.avg:>7.4f}) '
                        'Acc@5: {top5.val:>7.4f} ({top5.avg:>7.4f})'.format(
                            logger_name, i, steps_per_epoch, data_time=data_time_m,
                            batch_time=batch_time_m,
                            loss=losses_m, top1=top1_m, top5=top5_m))
        return top1_m.avg, top5_m.avg

    def _init_trainer(self):
        if self.last_train is None:
            raise RuntimeError('Cannot init trainer without knowing the size of training data')
        if isinstance(self.last_train, pd.DataFrame):
            train_size = len(self.last_train)
        elif isinstance(self.last_train, int):
            train_size = self.last_train
        else:
            raise ValueError("Unknown type of self.last_train: {}".format(type(self.last_train)))

        batch_size = self._cfg.train.batch_size
        self.batch_size = batch_size
        num_batches = train_size // batch_size

        base_lr = self._cfg.train.base_lr
        warmup_epochs = self._cfg.train.warmup_epochs
        decay_factor = self._cfg.train.decay_factor
        lr_decay_period = self._cfg.train.lr_decay_period

        if self._cfg.train.lr_decay_period > 0:
            lr_decay_epoch = list(range(lr_decay_period, self._cfg.train.epochs, lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in self._cfg.train.lr_decay_epoch.split(',')]

        if self._cfg.train.lr_schedule_mode == "step":
            lr_policy = lr_step_policy(base_lr=base_lr, steps=lr_decay_epoch,
                decay_factor=decay_factor, warmup_length=warmup_epochs, logger=self._logger)
        elif self._cfg.train.lr_schedule_mode == "cosine":
            lr_policy = lr_cosine_policy(base_lr=base_lr, warmup_length=warmup_epochs, epochs=self.epoch,
                end_lr=self._cfg.train.end_lr, logger=self._logger)
        elif self._cfg.train.lr_schedule_mode == "linear":
            lr_policy = lr_linear_policy(base_lr=base_lr, warmup_length=warmup_epochs, epochs=self.epochs,
                logger=self._logger)

        if self._optimizer is None:
            optimizer = optim.SGD(params=self.net.parameters(), lr=base_lr,
                                    momentum=self._cfg.train.momentum, weight_decay=self._cfg.train.weight_decay,
                                    nesterov=self._cfg.train.nesterov)
        else:
            optimizer = self._optimizer
            if isinstance(optimizer, str):
                try:
                    optimizer = get_optimizer(optimizer, lr=base_lr)
                except TypeError:
                    pass
        # init loss function
        loss = nn.CrossEntropyLoss
        if self._cfg.train.mixup:
            loss = lambda: NLLMultiLabelSmooth(self._cfg.train.mixup_alpha)
        elif self._cfg.train.label_smoothing:
            loss = lambda: LabelSmoothing(self._cfg.train.mixup_alpha)

        # amp trainng
        scaler = torch.cuda.amp.GradScaler(
            init_scale=self._cfg.train.static_loss_scale,
            growth_factor=2,
            backoff_factor=0.5,
            growth_interval=100 if self._cfg.train.dynamic_loss_scale else 1000000000,
            enabled=self._cfg.train.amp,
        )

        self.scaler = scaler
        self.lr_policy = lr_policy
        self.optimizer = optimizer
        self.criterion = loss().cuda()

    def _init_network(self, **kwargs):
        load_only = kwargs.get('load_only', False)
        if not self.num_class:
            raise ValueError('This is a classification problem and we are not able to create network when `num_class` is unknown. \
                It should be inferred from dataset or resumed from saved states.')
        assert len(self.classes) == self.num_class

        valid_gpus = []
        if self._cfg.gpus:
            valid_gpus = self._validate_gpus(self._cfg.gpus)
            if not valid_gpus:
                self._logger.warning(
                    'No gpu detected, fallback to cpu. You can ignore this warning if this is intended.')
            elif len(valid_gpus) != len(self._cfg.gpus):
                self._logger.warning(
                    f'Loaded on gpu({valid_gpus}), different from gpu({self._cfg.gpus}).')

        self.ctx = [int(i) for i in valid_gpus]

        # network
        if self._custom_net is None:
            model_name = self._cfg.img_cls.model_name.lower()
            input_size = self.input_size
            self.input_size = get_input_size(model_name)
        else:
            self._logger.debug('Custom network specified, ignore the model name in config...')
            self.net = copy.deepcopy(self._custom_net)
            model_name = ''
            self.input_size = input_size = self._cfg.train.input_size

        if input_size != self.input_size:
            self._logger.info(f'Change input size to {self.input_size}, given model type: {model_name}')

        use_pretrained = not load_only and self._cfg.img_cls.use_pretrained
        if model_name:
            self.net = init_network(model_name, num_class=self.num_class, pretrained=use_pretrained)

    def evaluate(self, val_data, metric_name=None):
        return self._evaluate(val_data, metric_name=metric_name)

    def _evaluate(self, val_data, metric_name=None):
        """Test on validation dataset."""
        return None, None

    def _predict_preprocess(self, x):
        resize = int(math.ceil(self.input_size / self._cfg.train.crop_ratio))
        return None

    def _predict(self, x, ctx_id=0, with_proba=False):
        return df

    def _get_feature_net(self):
        """Get the network slice for feature extraction only"""
        if hasattr(self, '_feature_net') and self._feature_net is not None:
            return self._feature_net
        self._feature_net = copy.copy(self.net)
        fc_layer_found = False
        for fc_name in ('output', 'fc'):
            fc_layer = getattr(self._feature_net, fc_name, None)
            if fc_layer is not None:
                fc_layer_found = True
                break
        if fc_layer_found:
            self._feature_net.register_child(nn.Identity(), fc_name)
            super(gluon.Block, self._feature_net).__setattr__(fc_name, nn.Identity())
            self.net.__setattr__(fc_name, fc_layer)
        else:
            raise RuntimeError('Unable to modify the last fc layer in network, (output, fc) expected...')
        return self._feature_net

    def _predict_feature(self, x, ctx_id=0):
        return df

    def _predict_proba(self, x, ctx_id=0):
        return df