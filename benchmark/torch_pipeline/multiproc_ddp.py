import argparse
import os
import random
import time
import warnings
import logging
import numpy as np
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.utils.data.distributed
import torch.multiprocessing as mp
import torch.utils.data

from autotorch.data import *
from autotorch.data.mixup import NLLMultiLabelSmooth, MixUpWrapper
from autotorch.data.smoothing import LabelSmoothing
from autotorch.models.model_zoo import get_model_list
from autotorch.models.network import init_network, get_input_size
from autotorch.models.common import EMA
from autotorch.optim.optimizers import get_optimizer
from autotorch.scheduler.lr_scheduler import *
from autotorch.utils.model import resum_checkpoint
from autotorch.training import ModelAndLoss, train_loop
import autogluon.core as ag
from test import test


def parse_args():
    parser = argparse.ArgumentParser(description='Model-based Asynchronous HPO')
    parser.add_argument('--data_name', default="", type=str, help='dataset name')
    parser.add_argument('--data_path', default="", type=str, help='path to dataset')
    parser.add_argument("--data-backend", metavar="BACKEND", default="pytorch", 
                        choices=DATA_BACKEND_CHOICES, 
                        help="data backend: " + " | ".join(DATA_BACKEND_CHOICES) + " (default: pytorch)",)
    parser.add_argument('--interpolation', metavar="INTERPOLATION", default="bilinear",
                        help="interpolation type for resizing images: bilinear, bicubic or triangular(DALI only)",)
    model_names = get_model_list()
    parser.add_argument('--model', metavar='MODEL', default='resnet18', choices=model_names,
                        help='model architecture: ' + ' | '.join(model_names) + ' (default: resnet18)')
    parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                        help='use pre-trained model')
    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument("--run-epochs", default=-1, type=int, metavar="N", 
                        help="run only N epochs, used for checkpointing runs",)
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument("--early-stopping-patience", default=-1, type=int, metavar="N",
                        help="early stopping after N epochs without validation accuracy improving",)
    parser.add_argument('--image-size', default=None, type=int, help="resolution of image")
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help="mini-batch size (default: 256) per gpu")
    parser.add_argument('--optimizer-batch-size', default=-1, type=int, metavar="N",
                        help="size of a total batch size, for simulating bigger batches using gradient accumulation",)
    parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--lr-schedule', default="step", type=str, metavar="SCHEDULE",
                        choices=["step", "linear", "cosine", "exponential"],
                        help="Type of LR schedule: {}, {}, {} , {}".format("step", "linear", "cosine", "exponential"),)
    parser.add_argument('--auto-step', default=True, type=bool, help="Use auto-step lr-schedule or not")                       
    parser.add_argument('--warmup', default=0, type=int, metavar="E", help="number of warmup epochs")
    parser.add_argument('--label-smoothing', default=0.0, type=float, metavar="S", help="label smoothing")
    parser.add_argument('--mixup', default=0.0, type=float, metavar="ALPHA", help="mixup alpha")
    parser.add_argument('--optimizer', default="sgd", type=str, choices=("sgd", "rmsprop"))
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M', help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument("--bn-weight-decay", action="store_true",
                        help="use weight_decay on batch normalization learnable parameters, (default: false)",)
    parser.add_argument('--rmsprop-alpha', default=0.9, type=float, help="value of alpha parameter in rmsprop optimizer (default: 0.9)",)
    parser.add_argument('--rmsprop-eps', default=1e-3, type=float, help="value of eps parameter in rmsprop optimizer (default: 1e-3)",)
    parser.add_argument('--nesterov', action="store_true", help="use nesterov momentum, (default: false)",)
    parser.add_argument('--use-ema', default=None, type=float, help="use EMA")
    parser.add_argument('--augmentation', type=str, default=None, 
                        choices=[None, "autoaugment", "original-mstd0.5", "rand-m9-n3-mstd0.5", "augmix-m5-w4-d2"], 
                        help="augmentation method",)
    parser.add_argument('--log_interval', default=10, type=int, metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
    parser.add_argument("--training-only", action="store_true", help="do not evaluate")
    parser.add_argument("--no-checkpoints", action="store_false", dest="save_checkpoints",
                        help="do not store any checkpoints, useful for benchmarking",)
    parser.add_argument("--checkpoint-filename", default="checkpoint.pth.tar", type=str)
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help="Static loss scale, positive power of 2 values can improve amp convergence.",)
    parser.add_argument('--dynamic-loss-scale', action="store_true",
                        help="Use dynamic loss scaling.  If supplied, this argument supersedes " + "--static-loss-scale.",)
    parser.add_argument("--memory-format", type=str, default="nchw", choices=["nchw", "nhwc"],
                        help="memory layout, nchw or nhwc",)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument('--dist-url', default='env://', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--gpu', default=None, type=int,
                        help='GPU id to use.')
    parser.add_argument('--multiprocessing-distributed', action='store_true',
                        help='Use multi-processing distributed training to launch '
                            'N processes per node, which has N GPUs. This is the '
                            'fastest way to use PyTorch for either single node or '
                            'multi node data parallel training')
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--output-dir', default="/home/yiran.wu/work_dirs/pytorch_model_benchmark", type=str,
                        help='output directory for model and log')
    args = parser.parse_args()
    return args


def main(args):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        if "WORLD_SIZE" in os.environ:
            args.local_rank = int(os.environ["LOCAL_RANK"])
            args.world_size = int(os.environ["WORLD_SIZE"])
        else:
            args.local_rank = 0

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, logger, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, logger, args)


def main_worker(gpu, ngpus_per_node, logger, args):
    global best_prec1
    best_prec1 = 0

    args.gpu = gpu
    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
    # set amp
    if args.static_loss_scale != 1.0:
        if not args.amp:
            print("Warning: if --amp is not used, static_loss_scale will be ignored.")

    if args.optimizer_batch_size < 0:
        batch_size_multiplier = 1
    else:
        tbs = args.world_size * args.batch_size

        if args.optimizer_batch_size % tbs != 0:
            print(
                "Warning: simulated batch size {} is not divisible by actual batch size {}".format(
                    args.optimizer_batch_size, tbs
                )
            )
        batch_size_multiplier = int(args.optimizer_batch_size / tbs)
        print("BSM: {}".format(batch_size_multiplier))

    start_epoch = 0
    # set the image_size
    image_size = (args.image_size
        if args.image_size is not None
        else get_input_size(args.model)
    )
    memory_format = (
        torch.channels_last if args.memory_format == "nhwc" else torch.contiguous_format
    )

    # Creat train losses
    loss = nn.CrossEntropyLoss
    if args.mixup > 0.0:
        loss = lambda: NLLMultiLabelSmooth(args.label_smoothing)
    elif args.label_smoothing > 0.0:
        loss = lambda: LabelSmoothing(args.label_smoothing)

    # Create data loaders
    if args.data_backend == "pytorch":
        get_train_loader = get_pytorch_train_loader_
        get_val_loader = get_pytorch_val_loader_
    elif args.data_backend == "dali-gpu":
        get_train_loader = get_dali_train_loader(dali_cpu=False)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "dali-cpu":
        get_train_loader = get_dali_train_loader(dali_cpu=True)
        get_val_loader = get_dali_val_loader()
    elif args.data_backend == "syntetic":
        get_val_loader = get_syntetic_loader
        get_train_loader = get_syntetic_loader
    else:
        print("Bad databackend picked")
        exit(1)

    # get data loaders
    train_loader, num_class = get_train_loader(
        args.data_path,
        "train",
        image_size,
        args.batch_size,
        args.mixup > 0.0,
        interpolation=args.interpolation,
        augmentation=args.augmentation,
        start_epoch=start_epoch,
        workers=args.workers,
    )
    if args.mixup != 0.0:
        train_loader = MixUpWrapper(args.mixup, train_loader)

    val_loader, _ = get_val_loader(
        args.data_path,
        "val",
        image_size,
        args.batch_size,
        False,
        interpolation=args.interpolation,
        workers=args.workers,
    )

    # model
    model = init_network(args.model, num_class, pretrained=args.pretrained)

    if not torch.cuda.is_available():
        print('using CPU, this will be slow')
    elif args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            #args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=0)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model, output_device=0)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
            model.features = torch.nn.DataParallel(model.features)
            model.cuda()
        else:
            model = torch.nn.DataParallel(model).cuda()

    # optionally resume from a checkpoint
    if args.resume is not None:
        model_state, model_state_ema, optimizer_state, start_epoch, best_prec1 = resum_checkpoint(args.resume)
    else:
        model_state = None
        model_state_ema = None
        optimizer_state = None
    
    # EMA
    if args.use_ema is not None:
        model_ema = deepcopy(model)
        ema = EMA(args.use_ema)
    else:
        model_ema = None
        ema = None

    # define loss function (criterion) and optimizer
    criterion = loss().cuda(args.gpu)

    # optimizer and lr_policy
    optimizer = get_optimizer(
        list(model.named_parameters()),
        args.lr,
        args=args,
        state=optimizer_state,
    )
    # lr policy
    if args.lr_schedule == "step":
        if args.auto_step:
            step_ratios = [0.6, 0.9]
            auto_steps = [int(ratio * args.epochs) for ratio in step_ratios]
            lr_policy = lr_step_policy(
                base_lr=args.lr, steps=auto_steps, decay_factor=0.1, warmup_length=args.warmup, logger=logger
            )
        else:
            lr_policy = lr_step_policy(
                base_lr=args.lr, steps=[30, 60, 80], decay_factor=0.1, warmup_length=args.warmup, logger=logger
            )
    elif args.lr_schedule == "cosine":
        lr_policy = lr_cosine_policy(
            base_lr=args.lr, warmup_length=args.warmup, epochs=args.epochs, end_lr=args.end_lr, logger=logger
        )
    elif args.lr_schedule == "linear":
        lr_policy = lr_linear_policy(base_lr=args.lr, warmup_length=args.warmup, epochs=args.epochs, logger=logger
        )
    elif args.lr_schedule == "exponential":
        lr_policy = lr_exponential_policy(base_lr=args.lr, warmup_length=args.warmup, epochs=args.epochs, logger=logger
        )
 
    scaler = torch.cuda.amp.GradScaler(
        init_scale=args.static_loss_scale,
        growth_factor=2,
        backoff_factor=0.5,
        growth_interval=100 if args.dynamic_loss_scale else 1000000000,
        enabled=args.amp,
    )

    if model_state is not None:
        model.load_model_state(model_state)

    # trining and eval
    train_loop(
        model,
        criterion,
        optimizer,
        scaler,
        lr_policy,
        train_loader,
        val_loader,
        num_class,
        logger=logger,
        use_amp=args.amp,
        batch_size_multiplier=batch_size_multiplier,
        start_epoch=start_epoch,
        end_epoch=min((start_epoch + args.run_epochs), args.epochs) if args.run_epochs != -1 else args.epochs,
        early_stopping_patience=args.early_stopping_patience,
        best_prec1=best_prec1,
        skip_training=args.evaluate,
        skip_validation=args.training_only,
        save_checkpoints=args.save_checkpoints and not args.evaluate,
        checkpoint_dir=args.output_dir,
        checkpoint_filename=args.checkpoint_filename,
    )
    print("Experiment ended")


if __name__ == '__main__':
    args = parse_args()
    task_name = args.data_name + '-' + args.model
    args.output_dir = os.path.join(args.output_dir, task_name)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    global logger
    logger = logging.getLogger('')
    filehandler = logging.FileHandler(os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    cudnn.benchmark = True

    main(args)
    args.resume = os.path.join(args.output_dir, "model_best.pth.tar")
    prec1 = test(args, logger)
    logger.info("Test Acc of Top1 is %s" % prec1)