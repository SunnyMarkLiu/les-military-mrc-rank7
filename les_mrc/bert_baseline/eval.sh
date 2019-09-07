#!/bin/bash
set -ex
DATA_DIR="/home/len/yingzq/dureader"
MODEL_DIR="/home/len/yingzq/pretrained-models/chinese_L-12_H-768_A-12"
RELOAD_MODEL_DIR="models/bert_finetuned_les"

python run_les.py \
    --model_type bert \
    --model_name_or_path $RELOAD_MODEL_DIR/pytorch_model.bin \
    --config_name $MODEL_DIR/bert_config.json \
    --tokenizer_name $MODEL_DIR/vocab.txt \
    --do_eval \
    --do_lower_case \
    --train_file $DATA_DIR/bert.tiny.json \
    --predict_file $DATA_DIR/bert.tiny.json \
    --output_dir $RELOAD_MODEL_DIR \
    --version_2_with_negative \
    --max_seq_length 512 \
    --max_answer_length 110 \
    --per_gpu_eval_batch_size 18 \
    --doc_stride 128 \
    --logging_steps 10 \
