import argparse
import os
import random
import logging
import time

import json
import numpy as np
from copy import deepcopy
import torch
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
from autotorch.models.common import EMA
from autotorch.optim.optimizers import get_optimizer
from autotorch.scheduler.lr_scheduler import *
from autotorch.utils.model import test_load_checkpoint, load_checkpoint
from autotorch.training import ModelAndLoss, validate
import autogluon.core as ag


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
    parser.add_argument('--model', metavar='MODEL', default='resnet18',
                        choices=model_names,
                        help='model architecture: ' +
                            ' | '.join(model_names) +
                            ' (default: resnet18)')
    parser.add_argument('-j', '--workers', type=int, default=4, metavar='N',
                        help='how many training processes to use (default: 1)')
    parser.add_argument('--image-size', default=None, type=int, help="resolution of image")
    parser.add_argument('-b', '--batch-size', default=256, type=int, metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--resume', default=None, type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--evaluate', dest='evaluate', action='store_true',
                        help='evaluate model on validation set')
    parser.add_argument('--use-ema', default=None, type=float, help="use EMA")
    parser.add_argument('--seed', type=int, default=42, metavar='S',
                        help='random seed (default: 42)')
    parser.add_argument('--amp', action='store_true', default=False,
                        help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
    parser.add_argument('--apex-amp', action='store_true', default=False,
                        help='Use NVIDIA Apex AMP mixed precision')
    parser.add_argument('--native-amp', action='store_true', default=False,
                        help='Use Native Torch AMP mixed precision')
    parser.add_argument('--static-loss-scale', type=float, default=1,
                        help="Static loss scale, positive power of 2 values can improve amp convergence.")
    parser.add_argument('--mixup', default=0.0, type=float, metavar="ALPHA", help="mixup alpha")
    parser.add_argument('--label-smoothing', default=0.0, type=float, metavar="S", help="label smoothing")

    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument('--world-size', default=-1, type=int,
                        help='number of nodes for distributed training')
    parser.add_argument('--rank', default=-1, type=int,
                        help='node rank for distributed training')
    parser.add_argument("--memory-format", type=str, default="nchw", choices=["nchw", "nhwc"],
                        help="memory layout, nchw or nhwc",)
    parser.add_argument('--output-dir', default="/home/yiran.wu/work_dirs/pytorch_model_benchmark", type=str,
                        help='output directory for model and log')
    parser.add_argument('--output_path', default="/home/yiran.wu/work_dirs/pytorch_model_benchmark", type=str,
                        help='output directory for model and log')
    args,_ = parser.parse_known_args()
    if args.output_path is not None:
        args.output_dir=args.output_path
    return args


def prepare_for_test(args):
    args.distributed = False
    if "WORLD_SIZE" in os.environ:
        args.distributed = int(os.environ["WORLD_SIZE"]) > 1
        args.local_rank = int(os.environ["LOCAL_RANK"])
    else:
        args.local_rank = 0

    args.gpu = 0
    args.world_size = 1

    if args.distributed:
        args.gpu = args.local_rank % torch.cuda.device_count()
        torch.cuda.set_device(args.gpu)
        if not torch.distributed.is_initialized():
            dist.init_process_group(backend="nccl", init_method="env://")
            args.world_size = torch.distributed.get_world_size()

    if args.seed is not None:
        print("Using seed = {}".format(args.seed))
        torch.manual_seed(args.seed + args.local_rank)
        torch.cuda.manual_seed(args.seed + args.local_rank)
        np.random.seed(seed=args.seed + args.local_rank)
        random.seed(args.seed + args.local_rank)

        def _worker_init_fn(id):
            np.random.seed(seed=args.seed + args.local_rank + id)
            random.seed(args.seed + args.local_rank + id)

    else:

        def _worker_init_fn(id):
            pass

    if args.static_loss_scale != 1.0:
        if not args.amp:
            print("Warning: if --amp is not used, static_loss_scale will be ignored.")

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
        get_train_loader = get_pytorch_train_loader
        get_val_loader = get_pytorch_val_loader
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

    test_loader, num_class = get_val_loader(
        args.data_path,
        "test",
        image_size,
        args.batch_size,
        False,
        interpolation=args.interpolation,
        workers=args.workers,
        # memory_format=memory_format,
    )

    # model
    model = init_network(args.model, num_class, pretrained=False)

    if args.distributed:
        # DistributedDataParallel will divide and allocate batch_size to all
        # available GPUs if device_ids are not set
        torch.cuda.set_device(args.gpu)
        model.cuda(args.gpu).to(memory_format=memory_format)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu],  output_device=args.gpu)
    else:
        model.cuda().to(memory_format=memory_format)

    # optionally resume from a checkpoint
    if args.resume is not None:
       model_state, model_state_ema, optimizer_state = test_load_checkpoint(args)
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

    # load mode state
    if model_state is not None:
        print("load model checkpoint")
        model.load_state_dict(model_state, strict=False)

    if (ema is not None) and (model_state_ema is not None):
        print("load ema")
        ema.load_state_dict(model_state_ema)

    # define loss function (criterion) and optimizer
    criterion = loss().cuda(args.gpu)

    return (model, criterion, test_loader, ema, model_ema, num_class)


def test(args, logger):
    model, criterion, test_loader, ema, model_ema, num_class = prepare_for_test(args)
    use_ema = (model_ema is not None) and (ema is not None)
    prec1 = validate(test_loader, model, criterion, num_class, logger, "Test-log", use_amp=args.amp)
    if use_ema:
        model_ema.load_state_dict({k.replace('module.', ''): v for k, v in ema.state_dict().items()})
        prec1 = validate(test_loader,  model, criterion, num_class, logger, "Test-log")
    return prec1


if __name__ == "__main__":
    args = parse_args()
    logger = logging.getLogger('')
    filehandler = logging.FileHandler(os.path.join(args.output_dir, 'summary.log'))
    streamhandler = logging.StreamHandler()
    logger.setLevel(logging.INFO)
    logger.addHandler(filehandler)
    logger.addHandler(streamhandler)
    cudnn.benchmark = True
    start_time = time.time()
    prec1 = test(args, logger)
    # (0.6832659840583801, 0.6190476190476191, 1.0, 8)
    logger.info("**"*100)
    logger.info("Test Acc of Top1 is %s" % prec1[1])
    logger.info("Test Acc of Top5 is %s" % prec1[2])
    end_time = time.time()
    test_time = end_time - start_time
    logger.info("Total time of test is {:7.1f} s".format(test_time))
    with open(os.path.join(args.output_path, "eval_result.json"), "w")as f:
        json.dump({
            "name": f"lenet-classification-tensorflow",
            "evaluation": [
                {
                    "key": "accuary",
                    "value": f"{prec1[1]}",
                    "type": "float",
                    "desc": "分类准确率TOP1"
                },
                {
                    "key": "accuary",
                    "value": f"{prec1[2]}",
                    "type": "float",
                    "desc": "分类准确率TOP5"
                },
                {
                    "key": "time_cost",
                    "value": f"{test_time}s",
                    "type": "string",
                    "desc": "评估耗时"
                }
            ]
        }, f)