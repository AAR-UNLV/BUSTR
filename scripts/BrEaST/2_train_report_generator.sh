#!/bin/bash

cd "$(dirname "$0")/../.."

dataset="BrEaST"
annotation="../../datasets/BREAST_Dataset"
base_dir="../../datasets/BREAST_Dataset"

version="v_final"
savepath="./save/$dataset/$version"

python -u - < train_report_generator.py \
    --dataset ${dataset} \
    --annotation ${annotation} \
    --version ${version} \
    --base_dir ${base_dir} \
    --batch_size 8 \
    --val_batch_size 16 \
    --vision_version v_final \
    --freeze_vm False \
    --vis_use_lora False \
    --llm_use_lora False \
    --savedmodel_path ${savepath} \
    --max_length 150 \
    --min_new_tokens 100 \
    --max_new_tokens 150 \
    --repetition_penalty 1.0 \
    --length_penalty 1.0 \
    --num_workers 1 \
    --devices 1 \
    --max_epochs 25 \
    --limit_val_batches 1.0 \
    --val_check_interval 1.0 \
    --device_name "3" \
    --num_sanity_val_steps 0 \
    "$@"
