#!/bin/bash
python3 -m torch.distributed.launch --nproc_per_node=4 torch_ddp.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "resnet18" \
--lr 0.005 \
--epochs 1 \
--batch-size 256 \
--pretrained \
--multiprocessing-distributed > ddpoutput.txt 2>&1 &

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

python main.py \
    --data_name hymenoptera \
    --data_path /media/robin/DATA/datatsets/image_data/shopee-iet/images/ \
    --output-dir //media/robin/DATA/datatsets/image_data/shopee-iet \
    --model resnet18 \
    --epochs 10 \
    --lr 0.01 \
    --batch-size 16 \
    --pretrained


python multiproc_ddp.py  \
    --data_path /media/robin/DATA/datatsets/image_data/shopee-iet/images/ \
    --output-dir //media/robin/DATA/datatsets/image_data/shopee-iet \
    --data-backend pytorch  \
    --lr 0.1  \
    --batch-size 32 \
    --model resnet18 \
    --pretrained \
    --epochs 1  \
    ----multiprocessing-distributed \
    --world-size 1 \
    --rank 0 


nohup python3 test.py \
--data_name "UKCarsDataset" \
--data_path "/data/AutoML_compete/UKCarsDataset/split" \
--model "resnetv2_50x1_bitm_in21k" \
--resume "/home/yiran.wu/work_dirs/pytorch_model_benchmark/UKCarsDataset-resnetv2_50x1_bitm_in21k/model_best.pth.tar" \
-b 256 > testoutput.txt 2>&1 &