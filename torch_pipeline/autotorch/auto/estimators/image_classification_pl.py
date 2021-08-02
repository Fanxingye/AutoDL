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
from torch.cuda.amp import autocast
from torch.autograd import Variable
from torchvision import transforms
import torch.nn.functional as F
from pytorch_lightning import LightningModule
from torchmetrics.classification.accuracy import Accuracy

from autotorch.models.network import init_network, get_input_size
from autotorch.utils.filesystem import try_import
from .default import ImageClassificationCfg
from .base_estimator import BaseEstimator, set_default
from ..data.dataset import TorchImageClassificationDataset
from ..data.dataloader import get_pytorch_train_loader, get_pytorch_val_loader

problem_type_constants = try_import(
    package='autogluon.core.constants',
    fromlist=['MULTICLASS', 'BINARY', 'REGRESSION'],
    message='Failed to import problem type constants from autogluon.core.')
MULTICLASS = problem_type_constants.MULTICLASS
BINARY = problem_type_constants.BINARY
REGRESSION = problem_type_constants.REGRESSION

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
    net : torch.nn.Module
        The custom network. If defined, the model name in config will be ignored so your
        custom network will be used for training rather than pulling it from model zoo.
    """
    Dataset = TorchImageClassificationDataset

    def __init__(self,
                 config,
                 logger=None,
                 reporter=None,
                 net=None,
                 optimizer=None,
                 problem_type=None):
        super(ImageClassificationEstimator, self).__init__(config,
                                                           logger=logger,
                                                           reporter=reporter,
                                                           name=None)
        if problem_type is None:
            problem_type = MULTICLASS
        self._problem_type = problem_type
        self.last_train = None
        self.input_size = self._cfg.train.input_size

        if net is not None:
            assert isinstance(
                net, torch.nn.Module
            ), f"given custom network {type(net)}, torch.nn.Module expected"
        self._custom_net = net
        self._feature_net = None

        if optimizer is not None:
            if isinstance(optimizer, str):
                pass
            else:
                assert isinstance(optimizer, torch.optim.Optimizer)
        self._optimizer = optimizer

    def _fit(self, train_data, val_data, time_limit=math.inf):
        tic = time.time()
        self._best_acc = -float('inf')
        self.epoch = 0
        self._time_elapsed = 0
        if max(self._cfg.train.start_epoch,
               self.epoch) >= self._cfg.train.epochs:
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
        if max(self._cfg.train.start_epoch,
               self.epoch) >= self._cfg.train.epochs:
            return {'time', self._time_elapsed}

        train_loader = get_pytorch_train_loader(
            data_dir=self._cfg.train.data_dir,
            batch_size=self.batch_size,
            num_workers=self._cfg.train.num_workers,
            input_size=self.input_size,
            crop_ratio=self._cfg.train.crop_ratio,
            data_augment=self._cfg.train.data_augment,
            train_dataset=train_data,
            one_hot=self._cfg.train.mixup)

        val_loader = get_pytorch_val_loader(
            data_dir=self._cfg.train.data_dir,
            batch_size=self.batch_size,
            num_workers=self._cfg.valid.num_workers,
            input_size=self.input_size,
            crop_ratio=self._cfg.train.crop_ratio,
            val_dataset=val_data)

        self._time_elapsed += time.time() - tic

        # ------------
        # model
        # ------------
        model = LitClassifier(cfg)

        # ------------
        # training
        # ------------
        trainer = Trainer.from_argparse_args(args)
        trainer.fit(model, train_loader, val_loader)

        # ------------
        # testing
        # ------------
        trainer.test(test_dataloaders=test_loader)

        return {'train'}


class LitClassifier(LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.model = init_network()
        # loss function
        self.criterion = torch.nn.CrossEntropyLoss()

        # use separate metric instance for train, val and test step
        # to ensure a proper reduction over the epoch
        self.train_accuracy = Accuracy()
        self.val_accuracy = Accuracy()
        self.test_accuracy = Accuracy()

    def step(self, batch: Any):
        x, y = batch
        logits = self.forward(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        return loss, preds, y

    def training_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)
        # log train metrics
        acc = self.train_accuracy(preds, targets)
        self.log("train/loss",
                 loss,
                 on_step=True,
                 on_epoch=True,
                 prog_bar=True)
        self.log("train/acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        # we can return here dict with any tensors
        # and then read it in some callback or in training_epoch_end() below
        # remember to always return loss from training_step, or else backpropagation will fail!
        return {"loss": loss, "preds": preds, "targets": targets}

    def train_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'train')

    def validation_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("val/loss",
                 loss,
                 on_step=False,
                 on_epoch=True,
                 prog_bar=False)
        self.log("val/acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def test_step(self, batch: Any, batch_idx: int):
        loss, preds, targets = self.step(batch)

        # log val metrics
        acc = self.val_accuracy(preds, targets)
        self.log("test/loss", loss, on_step=False, on_epoch=True)
        self.log("test/acc", acc, on_step=False, on_epoch=True)

        return {"loss": loss, "preds": preds, "targets": targets}

    def validation_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'val')

    def test_epoch_end(self, outputs):
        return self._eval_epoch_end(outputs, 'test')

    def _eval_epoch_end(self, outputs, prefix):
        """
        Called at the end of test/validation to aggregate outputs
        :param outputs: list of individual outputs of each validation step
        :return:
        """
        # if returned a scalar from validation_step, outputs is a list of tensor scalars
        # we return just the average in this case (if we want)
        # return torch.stack(outputs).mean()

        loss_mean = 0
        acc_mean = 0
        for output in outputs:
            loss = output[f'{prefix}_loss']

            # reduce manually when using dp
            if self.trainer.use_dp or self.trainer.use_ddp2:
                loss = torch.mean(loss)
            loss_mean += loss

            # reduce manually when using dp
            acc = output[f'{prefix}_acc']
            if self.trainer.use_dp or self.trainer.use_ddp2:
                acc = torch.mean(acc)

            acc_mean += acc

        loss_mean /= len(outputs)
        acc_mean /= len(outputs)
        tqdm_dict = {f'{prefix}_loss': loss_mean, f'{prefix}_acc': acc_mean}
        result = {
            'progress_bar': tqdm_dict,
            'log': tqdm_dict,
            f'{prefix}_loss': loss_mean
        }
        return result

    def configure_optimizers(self):
        """Choose what optimizers and learning-rate schedulers to use in your optimization.
        Normally you'd need one. But in the case of GANs or similar you might have multiple.

        See examples here:
            https://pytorch-lightning.readthedocs.io/en/latest/common/lightning_module.html#configure-optimizers
        """
        optimizer = optim.Adam(self.parameters(),
                               lr=self.hparams.learning_rate,
                               weight_decay=self.hparams.weight_decay)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=10)
        return [optimizer], [scheduler]

