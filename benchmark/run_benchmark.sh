#!/bin/bash

# RUN BENCHMARK
python benchmark.py \
    --data_path /media/robin/DATA/datatsets/image_data/hymenoptera/images/train \
    --dataset hymenoptera --output_path /home/robin/jianzh/automl/autodl/benchmark \
    --train_framework autokeras

python benchmark.py \
    --data_path /media/robin/DATA/datatsets/image_data/dog-breed-identification \
    --output_path /home/robin/jianzh/automl/autodl/benchmark \
    --dataset dog-breed-identification \
    --train_framework autogluon

python benchmark.py \
    --data_path /media/robin/DATA/datatsets/image_data/hymenoptera/split/train \
    --output_path /home/robin/jianzh/automl/autodl/benchmark \
    --report_path /home/robin/jianzh/automl/autodl/benchmark \
    --dataset  hymenoptera \
    --model_config  'default' \
    --batch-size 32 \
    --num_epochs 10 \
    --num_trials 1 \
    --train_framework autogluon 

## use docker 
python benchmark.py \
    --data_path /home/image_data/hymenoptera/split/train \
    --output_path /home/autodl/benchmark \
    --report_path /home/automl/autodl/benchmark \
    --dataset  hymenoptera \
    --model_config  'default_hpo' \
    --batch-size 16 \
    --num_epochs 10 \
    --num_trials 4 \
    --train_framework autogluon 


## on aiarts
python benchmark.py \
    --data_path /data/AutoML_compete/Flowers-Recognition/split/train \
    --output_path /data/autodl/benchmark \
    --report_path /data/autodl/benchmark \
    --dataset  Flowers-Recognition \
    --batch-size 32 \
    --num_epochs 10 \
    --model_config  'default' \
    --train_framework autotorch 


## autotorch
python multiproc_ddp.py --nproc_per_node 2 benchmark.py  \
    --data_path /data/AutoML_compete/Flowers-Recognition/split/train \
    --output_path /home/jianzheng.nie/autodl/benchmark/checkpoint \
    --report_path /home/jianzheng.nie/autodl/benchmark/checkpoint \
    --dataset  Flowers-Recognition \
    --batch-size 64  \
    --num_epochs 1 \
    --model_config  'default' \
    --train_framework autotorch


python -m torch.distributed.launch --nproc_per_node 2 benchmark.py  \
    --data_path /data/AutoML_compete/Flowers-Recognition/split/train \
    --output_path /home/jianzheng.nie/autodl/benchmark/checkpoint \
    --report_path /home/jianzheng.nie/autodl/benchmark/checkpoint \
    --dataset  Flowers-Recognition \
    --ngpus-per-trial 2 \
    --batch-size 64  \
    --num_epochs 1 \
    --model_config  'default' \
    --train_framework autotorch