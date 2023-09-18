#!/bin/bash

NUM_NODES=1
NUM_GPUS_PER_NODE=4

BATCH_SIZE=32
ACCUM_STEP=2

SAVE_PATH=output/5e-5_500epoch_chemdner_biobert_large

set -x
mkdir -p $SAVE_PATH
NCCL_P2P_DISABLE=1 python main.py \
    --data_path data \
    --save_path $SAVE_PATH \
    --train_file chemdner_combined_train_annotations.json\
    --valid_file chemdner_combined_eval_annotations.json\
    --roberta_checkpoint dmis-lab/biobert-large-cased-v1.1\
    --corpus chemdner\
    --lr 5e-5 \
    --epochs 500 --eval_per_epoch 5 \
    --warmup 0.02 \
    --batch_size $((BATCH_SIZE / NUM_GPUS_PER_NODE / ACCUM_STEP)) \
    --gradient_accumulation_steps ${ACCUM_STEP} \
    --do_val \
    --gpus $NUM_GPUS_PER_NODE \
    --cache_dir /Mounts/rbg-storage1/users/urop/vincentf/.local/bin\

