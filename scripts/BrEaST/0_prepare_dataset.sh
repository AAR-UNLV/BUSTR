#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/../.."

dataset="BrEaST"
dataset_path="../../datasets/"

python -u - < preprocessing/prepare_dataset.py \
    --dataset ${dataset} \
    --dataset_path ${dataset_path} \
    "$@"
