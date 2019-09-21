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
    --do_eval \
    --do_only_predict \
    --do_lower_case \
    --data_dir ${DATA_DIR} \
    --test_file test_r0.json \
    --output_dir models/${MODEL_COMMENT} \
    --max_seq_length 128 \
    --per_gpu_eval_batch_size=128   \
    --gradient_accumulation_steps 1 \
    --logging_steps 0 \
