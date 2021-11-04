#!/bin/bash

python benchmark.py \
    --data_path /media/robin/DATA/datatsets/image_data/hymenoptera/split/train \
    --output_path /home/robin/jianzh/automl/autodl/benchmark \
    --report_path /home/robin/jianzh/automl/autodl/benchmark \
    --dataset  hymenoptera \
    --model_config  'default' \
    --batch-size 32 \
    --num_epochs 1 \
    --num_trials 1 \
    --proxy \
    --train_framework autogluon 

python benchmark.py \
    --data_path /media/robin/DATA/datatsets/image_data/hymenoptera/split/train \
    --output_path /home/robin/jianzh/automl/autodl/benchmark \
    --report_path /home/robin/jianzh/automl/autodl/benchmark \
    --dataset  hymenoptera \
    --model_config  'default' \
    --batch-size  16 \
    --num_epochs 1 \
    --num_trials 1 \
    --proxy \
    --train_framework autotorch

## use docker 
python benchmark.py \
    --data_path /home/image_data/hymenoptera/split/train \
    --output_path /home/autodl/benchmark \
    --report_path /home/automl/autodl/benchmark \
    --dataset  hymenoptera \
    --model_config  'default_hpo' \
    --batch-size 64 \
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


## test pipeline
python test_pipeline.py --data_name  hymenoptera \
                        --data_path /media/robin/DATA/datatsets/image_data/hymenoptera \
                        --output_path /home/robin/jianzh/automl/autodl/torch_pipeline  \
                        --device_limit 1