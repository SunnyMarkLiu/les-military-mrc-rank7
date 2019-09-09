#!/bin/bash
set -ex
DATA_DIR="/home/lq/projects/Research/Reading-Comprehension/les-military-mrc/input/mrc_dataset"
MODEL_DIR="/home/lq/projects/deep_learning/yingzq/pretrained_weights/chinese_wwm_pytorch"
RELOAD_MODEL_DIR="models/bert_finetuned_les_wwm/checkpoint-best"

python run_les.py \
    --cuda_devices 3 \
    --model_type bert \
    --model_name_or_path $RELOAD_MODEL_DIR/pytorch_model.bin \
    --config_name $MODEL_DIR/bert_config.json \
    --tokenizer_name $MODEL_DIR/vocab.txt \
    --do_eval \
    --do_lower_case \
    --predict_file $DATA_DIR/les.dev.json \
    --output_dir $RELOAD_MODEL_DIR \
    --version_2_with_negative \
    --max_seq_length 512 \
    --max_query_length 64 \
    --max_answer_length 110 \
    --per_gpu_eval_batch_size 48 \
    --doc_stride 128 \
    --logging_steps 0 \
