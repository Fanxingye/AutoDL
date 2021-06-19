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
    --batch-size 8 \
    --num_epochs 1 \
    --train_framework autogluon 

## on aiarts
python benchmark.py \
    --data_path /data/AutoML_compete/Flowers-Recognition/split/train \
    --output_path /home/yiran.wu/work_dirs/autodl_benchmark \
    --report_path /home/yiran.wu/work_dirs/autodl_benchmark \
    --dataset  Flowers-Recognition \
    --batch-size 32 \
    --num_epochs 10 \
    --model_config  'default' \
    --train_framework autogluon 