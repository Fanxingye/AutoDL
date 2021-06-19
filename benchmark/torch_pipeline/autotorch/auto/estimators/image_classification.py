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
import torch.utils.data.distributed
from autotorch.data import *
from autotorch.data.mixup import NLLMultiLabelSmooth, MixUpWrapper
from autotorch.data.smoothing import LabelSmoothing
from autotorch.models.model_zoo import get_model_list
from autotorch.models.network import init_network, get_input_size
from autotorch.optim.optimizers import get_optimizer
from autotorch.scheduler.lr_scheduler import *
from autotorch.utils.model import resum_checkpoint
from autotorch.training import ModelAndLoss, train_loop

import autogluon.core as ag
from .base_estimator import BaseEstimator
from .default import ImageClassificationCfg
from ..data.dataset import TorchImageClassificationDataset
from gluoncv.auto.estimators.image_classification.utils import EarlyStopperOnPlateau
from gluoncv.utils.filesystem import try_import
problem_type_constants = try_import(package='autogluon.core.constants',
                                    fromlist=['MULTICLASS', 'BINARY', 'REGRESSION'],
                                    message='Failed to import problem type constants from autogluon.core.')
MULTICLASS = problem_type_constants.MULTICLASS
BINARY = problem_type_constants.BINARY
REGRESSION = problem_type_constants.REGRESSION

__all__ = ['ImageClassificationEstimator']


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
    net : mx.gluon.Block
        The custom network. If defined, the model name in config will be ignored so your
        custom network will be used for training rather than pulling it from model zoo.
    """
    Dataset = TorchImageClassificationDataset

    def __init__(self, config, logger=None, reporter=None, net=None, optimizer=None, problem_type=None):
        super(ImageClassificationEstimator, self).__init__(config, logger=logger, reporter=reporter, name=None)
        if problem_type is None:
            problem_type = MULTICLASS
        self._problem_type = problem_type
        self.last_train = None
        self.input_size = self._cfg.train.input_size
        self._feature_net = None

        if optimizer is not None:
            if isinstance(optimizer, str):
                pass
            else:
                assert isinstance(optimizer, optim)
        self._optimizer = optimizer

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
        self._init_trainer()
        self._time_elapsed += time.time() - tic
        return self._resume_fit(train_data, val_data, time_limit=time_limit)

    def _resume_fit(self, train_data, val_data, time_limit=math.inf):
        tic = time.time()
        if max(self._cfg.train.start_epoch, self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}
        if self._problem_type != REGRESSION and (not self.classes or not self.num_class):
            raise ValueError('This is a classification problem and we are not able to determine classes of dataset')

        num_workers = self._cfg.train.num_workers

        train_dataset = train_data.to_pytorch()
        val_dataset = val_data.to_pytorch()
        train_loader, val_loader, self.batch_fn = get_data_loader(self._cfg.train.data_dir,
                                                                    self.batch_size, num_workers,
                                                                    self.input_size,
                                                                    self._cfg.train.crop_ratio,
                                                                    train_dataset=train_dataset,
                                                                    val_dataset=val_dataset)
        self._time_elapsed += time.time() - tic
        return self._train_loop(train_loader, val_loader, time_limit=time_limit)

    def _train_loop(self, train_data, val_data, time_limit=math.inf):
        return None

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
        lr_decay_period = self._cfg.train._lr_decay_period

        if self._cfg.train.lr_decay_period > 0:
            lr_decay_epoch = list(range(lr_decay_period, self._cfg.train.epochs, lr_decay_period))
        else:
            lr_decay_epoch = [int(i) for i in self._cfg.train.lr_decay_epoch.split(',')]
            
        if self._cfg.lr_schedule_mode = "step":
            lr_scheuler = lr_step_policy(base_lr=base_lr, steps=lr_decay_epoch,
                decay_factor=decay_factor, warmup_length=warmup_epochs, logger=logger)
        elif self._cfg.lr_schedule == "cosine":
            lr_policy = lr_cosine_policy(base_lr=base_lr, warmup_length=warmup_epochs, epochs=self.epoch, 
                end_lr=self._cfg.train.end_lr, logger=logger)
        elif self._cfg.lr_schedule == "linear":
            lr_policy = lr_linear_policy(base_lr=base_lr, warmup_length=warmup_epochs, epochs=self.epochs, 
                logger=logger)

        if self._optimizer is None:
            optimizer = optim.SGD
            optimizer_params = {'wd': self._cfg.train.wd,
                                'momentum': self._cfg.train.momentum,
                                'lr_scheduler': lr_scheduler}


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


class ImageListDataset(Dataset):
    """An internal image list dataset for batch predict"""
    def __init__(self, imlist, fn):
        self._imlist = imlist
        self._fn = fn

    def __getitem__(self, idx):
        img = self._fn(self._imlist[idx])[0]
        return img

    def __len__(self):
        return len(self._imlist)


