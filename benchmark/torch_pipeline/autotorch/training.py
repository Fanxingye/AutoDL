import math
import time

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from autotorch.models.common import EMA
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast
from autotorch.utils.model import reduce_tensor, save_checkpoint
from autotorch.utils.time_handler import TimeoutHandler
from autotorch.utils.metrics import AverageMeter, accuracy


class ModelAndLoss(nn.Module):
    def __init__(
        self,
        model,
        loss,
        cuda=True,
        memory_format=torch.contiguous_format,
    ):
        super(ModelAndLoss, self).__init__()

        if cuda:
            model = model.cuda().to(memory_format=memory_format)

        # define loss function (criterion) and optimizer
        criterion = loss()

        if cuda:
            criterion = criterion.cuda()

        self.model = model
        self.loss = criterion

    def forward(self, data, target):
        output = self.model(data)
        loss = self.loss(output, target)

        return loss, output

    def sync_bn(self):
        self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)

    def distributed(self, gpu_id):
        self.model = DDP(self.model, device_ids=[gpu_id], output_device=gpu_id)

    def load_model_state(self, state):
        if state is not None:
            self.model.load_state_dict(state, strict=False)


def get_train_step(model, criterion, optimizer, scaler, use_amp=False, batch_size_multiplier=1, top_k=1):
    def _step(input, target, optimizer_step=True):
        input_var = Variable(input)
        target_var = Variable(target)

        with autocast(enabled=use_amp):
            output = model(input_var)
            loss = criterion(output, target_var)

            loss /= batch_size_multiplier
            prec1, prec5 = accuracy(output.data, target, topk=(1, min(top_k, 5)))
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

    return _step


def train(
    train_loader,
    model,
    criterion,
    optimizer,
    scaler,
    lr_scheduler,
    num_class,
    logger,
    epoch,
    timeout_handler,
    ema=None,
    use_amp=False,
    batch_size_multiplier=1,
    log_interval=1
):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')
    top1_m = AverageMeter('Acc@1', ':6.2f')
    top5_m = AverageMeter('Acc@5', ':6.2f')

    interrupted = False
    step = get_train_step(
        model,
        criterion,
        optimizer,
        scaler=scaler,
        use_amp=use_amp,
        batch_size_multiplier=batch_size_multiplier,
        top_k=num_class
    )

    model.train()
    optimizer.zero_grad()
    steps_per_epoch = len(train_loader)
    data_iter = enumerate(train_loader)
    end = time.time()
    for i, (input, target) in data_iter:
        input = input.cuda()
        target = target.cuda()

        bs = input.size(0)
        lr_scheduler(optimizer, i, epoch)
        data_time = time.time() - end

        optimizer_step = ((i + 1) % batch_size_multiplier) == 0
        loss, prec1, prec5 = step(input, target, optimizer_step=optimizer_step)
        if ema is not None:
            ema(model, epoch*steps_per_epoch+i)

        it_time = time.time() - end
        batch_time_m.update(it_time)
        data_time_m.update(data_time)
        losses_m.update(loss.item(), bs)
        top1_m.update(prec1.item(), bs)
        top5_m.update(prec5.item(), bs)

        end = time.time()
        if ((i+1) % 20 == 0) and timeout_handler.interrupted:
            time.sleep(5)
            interrupted = True
            break

        if (i % log_interval == 0) or (i == steps_per_epoch - 1):
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
    return interrupted


def get_val_step(model, criterion, use_amp=False, top_k=1):
    def _step(input, target):
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

    return _step


def validate(
    val_loader,
    model,
    criterion,
    num_class,
    logger,
    logger_name,
    use_amp=False,
    log_interval=10
):
    batch_time_m = AverageMeter('Time', ':6.3f')
    data_time_m = AverageMeter('Data', ':6.3f')
    losses_m = AverageMeter('Loss', ':.4e')
    top1_m = AverageMeter('Acc@1', ':6.2f')
    top5_m = AverageMeter('Acc@5', ':6.2f')

    step = get_val_step(model, criterion, use_amp=use_amp, top_k=num_class)
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

        if (i % log_interval == 0) or (i == steps_per_epoch - 1):
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
    return top1_m.avg


def train_loop(
    model,
    criterion,
    optimizer,
    scaler,
    lr_scheduler,
    train_loader,
    val_loader,
    num_class,
    logger,
    ema=None,
    model_ema=None,
    use_amp=False,
    batch_size_multiplier=1,
    best_prec1=0,
    start_epoch=0,
    end_epoch=0,
    early_stopping_patience=-1,
    skip_training=False,
    skip_validation=False,
    save_checkpoints=True,
    checkpoint_dir="./",
    checkpoint_filename="checkpoint.pth.tar",
):
    prec1 = -1
    use_ema = (model_ema is not None) and (ema is not None)

    if early_stopping_patience > 0:
        epochs_since_improvement = 0
    
    print(f"RUNNING EPOCHS FROM {start_epoch} TO {end_epoch}")
    with TimeoutHandler() as timeout_handler:
        interrupted = False
        for epoch in range(start_epoch, end_epoch):
            if not skip_training:
                interrupted = train(
                    train_loader,
                    model,
                    criterion,
                    optimizer,
                    scaler,
                    lr_scheduler,
                    num_class,
                    logger,
                    epoch,
                    timeout_handler,
                    ema=ema,
                    use_amp=use_amp,
                    batch_size_multiplier=batch_size_multiplier,
                    log_interval=10
                )

            if not skip_validation:
                prec1 = validate(
                    val_loader,
                    model,
                    criterion,
                    num_class,
                    logger,
                    "Val-log",
                    use_amp=use_amp,
                )
                if use_ema:
                    model_ema.load_state_dict({k.replace('module.', ''): v for k, v in ema.state_dict().items()})
                    prec1 = validate(
                        val_loader,
                        criterion,
                        model_ema,
                        num_class,
                        logger,
                        "Val-log"
                    )

                if prec1 > best_prec1:
                    is_best = True
                    best_prec1 = prec1
                else:
                    is_best = False
            else:
                is_best = True
                best_prec1 = 0

            if save_checkpoints and (
                not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0
            ):
                checkpoint_state = {
                    "epoch": epoch + 1,
                    "state_dict": model.state_dict(),
                    "best_prec1": best_prec1,
                    "optimizer": optimizer.state_dict(),
                }
                if use_ema:
                    checkpoint_state["state_dict_ema"] = ema.state_dict()

                save_checkpoint(
                    checkpoint_state,
                    is_best,
                    checkpoint_dir=checkpoint_dir,
                    filename=checkpoint_filename,
                )
            if early_stopping_patience > 0:
                if not is_best:
                    epochs_since_improvement += 1
                else:
                    epochs_since_improvement = 0
                if epochs_since_improvement >= early_stopping_patience:
                    break
            if interrupted:
                break