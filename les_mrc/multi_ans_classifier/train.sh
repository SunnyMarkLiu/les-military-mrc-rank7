#!/bin/bash
set -ex
DATA_DIR="/home/lq/projects/Research/Reading-Comprehension/les-military-mrc/input/mrc_dataset_test"
MODEL_DIR="/home/lq/projects/deep_learning/yingzq/pretrained_weights/chinese_wwm_pytorch"
MODEL_COMMENT="multi_ans_classifier"

python ./examples/run_glue.py \
    --cuda_devices 1 \
    --model_type bert \
    --model_name_or_path ${MODEL_DIR}/pytorch_model.bin \
    --config_name ${MODEL_DIR}/bert_config.json \
    --tokenizer_name ${MODEL_DIR}/vocab.txt \
    --task_name les-multi-ans \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --train_file ${DATA_DIR}/train_max_content_len_1000.json \
    --dev_file ${DATA_DIR}/dev.json \
    --output_dir models/${MODEL_COMMENT} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=8   \
    --per_gpu_train_batch_size=8   \
    --learning_rate 2e-5 \
    --num_train_epochs 3.0 \
    --gradient_accumulation_steps 1 \
    --logging_steps 100 \
    --save_steps 8000 \
    --eval_steps 4000 \
