'efficientnet_b4',
'efficientnet_b2',
'vit_base_r50_s16_384',
'resnetv2_50x1_bitm_in21k',
'resnetv2_101x1_bitm_in21k',
'swin_base_patch4_window7_224_in22k',

### Flowers-Recognition ###
nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "efficientnet_b4" \
--lr 0.005 \
--epochs 30 \
--batch-size 8 \
--early-stopping-patience 10 \
--pretrained \
--multiprocessing-distributed > outputlog/efficientnet_b4_Flowers.txt 2>&1 &

sleep 30m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "efficientnet_b2" \
--lr 0.005 \
--epochs 10 \
--batch-size 16 \
--early-stopping-patience 10 \
--pretrained \
--multiprocessing-distributed > outputlog/efficientnet_b2_Flowers.txt 2>&1 &

sleep 60m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "vit_base_r50_s16_384" \
--lr 0.005 \
--epochs 30 \
--batch-size 8 \
--early-stopping-patience 10 \
--pretrained \
--multiprocessing-distributed > outputlog/vit_base_r50_s16_384_Flowers.txt 2>&1 &

sleep 90m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--early-stopping-patience 10 \
--pretrained \
--multiprocessing-distributed > outputlog/resnetv2_50x1_bitm_in21k_Flowers.txt 2>&1 &

sleep 110m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--augmentation "original-mstd0.5" \
--early-stopping-patience 10 \
--pretrained \
--multiprocessing-distributed > outputlog/resnetv2_50x1_bitm_in21k_Flowers_aug.txt 2>&1 &

sleep 130m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "resnetv2_101x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--early-stopping-patience 10 \
--pretrained \
--multiprocessing-distributed > outputlog/resnetv2_101x1_bitm_in21k_Flowers.txt 2>&1 &

sleep 160m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Flowers-Recognition" \
--data_path "/data/AutoML_compete/Flowers-Recognition/split" \
--model "swin_base_patch4_window7_224_in22k" \
--lr 0.005 \
--epochs 30 \
--batch-size 32 \
--early-stopping-patience 10 \
--pretrained \
--multiprocessing-distributed > outputlog/swin_base_patch4_window7_224_in22k_Flowers.txt 2>&1 &


### Store-type-recognition ###
sleep 190m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Store-type-recognition" \
--data_path "/data/AutoML_compete/Store-type-recognition/split" \
--model "efficientnet_b4" \
--lr 0.005 \
--epochs 30 \
--batch-size 8 \
--pretrained \
--multiprocessing-distributed > outputlog/efficientnet_b4_Store.txt 2>&1 &

sleep 220m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Store-type-recognition" \
--data_path "/data/AutoML_compete/Store-type-recognition/split" \
--model "efficientnet_b2" \
--lr 0.005 \
--epochs 30 \
--batch-size 32 \
--pretrained \
--multiprocessing-distributed > outputlog/efficientnet_b2_Store.txt 2>&1 &

sleep 250m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Store-type-recognition" \
--data_path "/data/AutoML_compete/Store-type-recognition/split" \
--model "vit_base_r50_s16_384" \
--lr 0.005 \
--epochs 30 \
--batch-size 8 \
--pretrained \
--multiprocessing-distributed > outputlog/vit_base_r50_s16_384_Store.txt 2>&1 &

sleep 280m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Store-type-recognition" \
--data_path "/data/AutoML_compete/Store-type-recognition/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--pretrained > outputlog/resnetv2_50x1_bitm_in21k_Store.txt 2>&1 &

sleep 300m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Store-type-recognition" \
--data_path "/data/AutoML_compete/Store-type-recognition/split" \
--model "resnetv2_101x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--pretrained > outputlog/resnetv2_101x1_bitm_in21k_Store.txt 2>&1 &

sleep 330m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Store-type-recognition" \
--data_path "/data/AutoML_compete/Store-type-recognition/split" \
--model "swin_base_patch4_window7_224_in22k" \
--lr 0.005 \
--epochs 30 \
--batch-size 32 \
--pretrained > outputlog/swin_base_patch4_window7_224_in22k_Store.txt 2>&1 &


### National-Data-Science-Bowl ###
sleep 640m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "National-Data-Science-Bowl" \
--data_path "/data/AutoML_compete/datasciencebowl/split" \
--model "efficientnet_b4" \
--lr 0.005 \
--epochs 30 \
--batch-size 8 \
--pretrained > outputlog/efficientnet_b4_datasciencebowl.txt 2>&1 &

sleep 740m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "National-Data-Science-Bowl" \
--data_path "/data/AutoML_compete/datasciencebowl/split" \
--model "efficientnet_b2" \
--lr 0.005 \
--epochs 30 \
--batch-size 32 \
--pretrained > outputlog/efficientnet_b2_datasciencebowl.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "National-Data-Science-Bowl" \
--data_path "/data/AutoML_compete/datasciencebowl/split" \
--model "vit_base_r50_s16_384" \
--lr 0.005 \
--epochs 30 \
--batch-size 8 \
--pretrained > outputlog/vit_base_r50_s16_384_datasciencebowl.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "National-Data-Science-Bowl" \
--data_path "/data/AutoML_compete/datasciencebowl/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 64 \
--pretrained > outputlog/resnetv2_50x1_bitm_in21k_datasciencebowl.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "National-Data-Science-Bowl" \
--data_path "/data/AutoML_compete/datasciencebowl/split" \
--model "resnetv2_101x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 32 \
--pretrained > outputlog/resnetv2_101x1_bitm_in21k_datasciencebowl.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "National-Data-Science-Bowl" \
--data_path "/data/AutoML_compete/datasciencebowl/split" \
--model "swin_base_patch4_window7_224_in22k" \
--lr 0.005 \
--epochs 1 \
--batch-size 32 \
--pretrained > outputlog/swin_base_patch4_window7_224_in22k_datasciencebowl.txt 2>&1 &

### APTOS-2019-Blindness-Detection ###
sleep 360m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "efficientnet_b4" \
--lr 0.005 \
--epochs 30 \
--batch-size 8 \
--early-stopping-patience 10 \
--pretrained > outputlog/efficientnet_b4_aptos.txt 2>&1 &

sleep 400m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "efficientnet_b2" \
--lr 0.005 \
--epochs 1 \
--batch-size 32 \
--early-stopping-patience 10 \
--pretrained > outputlog/efficientnet_b2_aptos.txt 2>&1 &

sleep 440m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "vit_base_r50_s16_384" \
--lr 0.005 \
--epochs 30 \
--batch-size 8 \
--early-stopping-patience 10 \
--pretrained > outputlog/vit_base_r50_s16_384_aptos.txt 2>&1 &

sleep 480m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--early-stopping-patience 10 \
--pretrained > outputlog/resnetv2_50x1_bitm_in21k_aptos.txt 2>&1 &

sleep 520m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--augmentation "original-mstd0.5" \
--early-stopping-patience 10 \
--pretrained > outputlog/resnetv2_50x1_bitm_in21k_aptos_aug.txt 2>&1 &

sleep 560m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "resnetv2_101x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--early-stopping-patience 10 \
--pretrained > outputlog/resnetv2_101x1_bitm_in21k_aptos.txt 2>&1 &

sleep 600m; nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "swin_base_patch4_window7_224_in22k" \
--lr 0.005 \
--epochs 1 \
--batch-size 32 \
--early-stopping-patience 10 \
--pretrained > outputlog/swin_base_patch4_window7_224_in22k_aptos.txt 2>&1 &

### data aug ###

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "UKCarsDataset" \
--data_path "/data/AutoML_compete/UKCarsDataset/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--optimizer-batch-size 32 \
--pretrained > outputlog/resnetv2_50x1_bitm_in21k_UKCarsDataset.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 torch_ddp.py \
--data_name "UKCarsDataset" \
--data_path "/data/AutoML_compete/UKCarsDataset/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.005 \
--epochs 30 \
--batch-size 16 \
--pretrained \
--multiprocessing-distributed >> outputlog/resnetv2_50x1_bitm_in21k_UKCarsDataset_ddp.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "UKCarsDataset" \
--data_path "/data/AutoML_compete/UKCarsDataset/split" \
--model "resnet50" \
--lr 0.01 \
--epochs 50 \
--batch-size 32 \
--pretrained \
--augmentation 'autoaugment' \
--early-stopping-patience 10 \
--optimizer-batch-size 64  >> outputlog/resnet50_UKCarsDataset.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Cassava-Leaf-Disease" \
--data_path "/data/AutoML_compete/cassava-leaf-diease/split" \
--model "resnet50" \
--lr 0.01 \
--epochs 50 \
--batch-size 32 \
--pretrained \
--augmentation 'autoaugment' \
--early-stopping-patience 10 \
--optimizer-batch-size 64  >> outputlog/resnet50_Cassava-Leaf-Disease.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "resnet50" \
--lr 0.01 \
--epochs 50 \
--batch-size 32 \
--early-stopping-patience 10 \
--pretrained \
--optimizer-batch-size 64 > outputlog/resnet50_aptos.txt 2>&1 &

sleep 40m; 
nohup python -m torch.distributed.launch --nproc_per_node=4 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.003 \
--epochs 40 \
--batch-size 16 \
--early-stopping-patience 10 \
--augmentation 'autoaugment' \
--pretrained \
--optimizer-batch-size 64 >> outputlog/resnetv2_50x1_bitm_in21k_aptos_aug.txt 2>&1 &

nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "Dog-Breed-Identification" \
--data_path "/data/AutoML_compete/dog-breed-identification/split" \
--model "resnetv2_50x1_bitm_in21k" \
--lr 0.01 \
--epochs 30 \
--batch-size 16 \
--early-stopping-patience 10 \
--pretrained \
--optimizer-batch-size 32 > outputlog/resnetv2_50x1_bitm_in21k_Dog-Breed.txt 2>&1 &


nohup python3 -m torch.distributed.launch --nproc_per_node=2 main.py \
--data_name "APTOS-2019-Blindness-Detection" \
--data_path "/data/AutoML_compete/aptos2019-blindness-detection/split" \
--model "swin_base_patch4_window7_224_in22k" \
--lr 0.01 \
--epochs 50 \
--batch-size 32 \
--early-stopping-patience 10 \
--pretrained \
--optimizer-batch-size 64 > outputlog/swin_base_patch4_window7_224_in22k_aptos.txt 2>&1 &