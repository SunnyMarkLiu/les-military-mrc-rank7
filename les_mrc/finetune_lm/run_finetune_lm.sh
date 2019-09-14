#!/usr/bin/env bash

set -ex
DATA_DIR="/home/lq/Research/Reading-Comprehension/les-military-mrc/input/military_corpus/finetune_lm_corpus.txt"
MODEL_DIR="/home/lq/Research/Reading-Comprehension/pretrained_weights/chinese_wwm_pytorch"

python pregenerate_training_data.py \
    --train_corpus ${DATA_DIR}
