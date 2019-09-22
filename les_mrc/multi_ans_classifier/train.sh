#!/bin/bash
set -ex
DATA_DIR="/home/lq/Research/Reading-Comprehension/les-military-mrc/input/mrc_dataset_0912"
MODEL_DIR="/home/lq/Research/Reading-Comprehension/pretrained_weights/chinese_wwm_pytorch"
MODEL_COMMENT="les_multi_ans_classifier"

python run_glue.py \
    --cuda_devices 2,3 \
    --model_type bert \
    --model_name_or_path ${MODEL_DIR}/pytorch_model.bin \
    --config_name ${MODEL_DIR}/bert_config.json \
    --tokenizer_name ${MODEL_DIR}/vocab.txt \
    --task_name les-multi-ans \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --train_file train_max_content_len_1000.json \
    --dev_file dev.json \
    --output_dir models/${MODEL_COMMENT} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=48   \
    --per_gpu_train_batch_size=48   \
    --learning_rate 2e-5 \
    --warmup_steps 0 \
    --num_train_epochs 3.0 \
    --gradient_accumulation_steps 1 \
    --logging_steps 100 \
    --save_steps 2000 \
    --eval_steps 300 \
    --overwrite_cache \
