#!/bin/bash
python -m torch.distributed.run --nnodes=1 --nproc_per_node=2 --rdzv_backend=c10d main.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--output-dir "/home/jianzheng.nie/autodl/benchmark/torch_pipeline/checkpoint/bb" \
--model "resnet18" \
--epochs 2 \
--lr 0.01 \
--epochs 10 \
--batch-size 256 \
--pretrained 

## local machine
python -m torch.distributed.launch --nproc_per_node=1 main.py \
--data_name hymenoptera \
--data_path /media/robin/DATA/datatsets/image_data/hymenoptera/split/ \
--output-dir /media/robin/DATA/datatsets/image_data/hymenoptera \
--model resnet18 \
--epochs 10 \
--lr 0.01 \
--batch-size 16 \
--pretrained > output.txt 2>&1 &

python -m torch.distributed.launch --nproc_per_node=4 test.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "resnet18" \
--resume "/home/yiran.wu/work_dirs/pytorch_model_benchmark/Flowers-Recognition-resnet18/model_best.pth.tar" \
-b 16