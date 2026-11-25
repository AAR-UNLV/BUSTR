#!/bin/bash

cd "$(dirname "$0")/../.."

dataset="BrEaST"
annotation="../../datasets/BREAST_Dataset" 
base_dir="../../datasets/BREAST_Dataset"

version="v_final"
savepath="./save_vision/$dataset/$version"

python -u - < train_vision_encoder.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --base_dir ${base_dir} \
    --batch_size 8 \
    --is_vision True \
    --freeze_vm False \
    --vis_use_lora False \
    --val_batch_size 16 \
    --savedmodel_path ${savepath} \
    --num_workers 1 \
    --devices 1 \
    --max_epochs 100 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --device_name "6" \
    --num_sanity_val_steps 0 \
    "$@"