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


nohup python3 -m torch.distributed.launch --nproc_per_node=4 main.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "resnet18" \
--lr 0.005 \
--epochs 1 \
--batch-size 256 \
--pretrained \
--multiprocessing-distributed > ddpoutput.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "resnetv2_101x3_bitm" \
--lr 0.005 \
--epochs 50 \
--batch-size 32 \
--pretrained \
--multiprocessing-distributed > output.txt 2>&1 &


nohup python3 test.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "efficientnet_b4" \
--resume "/home/yiran.wu/work_dirs/pytorch_model_benchmark/Flowers-Recognitionefficientnet_b4/model_best.pth.tar" \
-b 8 > ddpoutput.txt 2>&1 &


nohup python3 test.py \
--data_name "UKCarsDataset" \
--data_path "/data/AutoML_compete/UKCarsDataset/split" \
--model "resnetv2_50x1_bitm_in21k" \
--resume "/home/yiran.wu/work_dirs/pytorch_model_benchmark/UKCarsDataset-resnetv2_50x1_bitm_in21k/model_best.pth.tar" \
-b 16 > testoutput.txt 2>&1 &