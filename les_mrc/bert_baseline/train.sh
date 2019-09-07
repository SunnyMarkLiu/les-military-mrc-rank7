#!/bin/bash
set -ex
DATA_DIR="/home/len/yingzq/les/mrc_dataset"
MODEL_DIR="/home/len/yingzq/pretrained-models/chinese_L-12_H-768_A-12"

python run_les.py \
    --model_type bert \
    --model_name_or_path $MODEL_DIR/pytorch_model.bin \
    --config_name $MODEL_DIR/bert_config.json \
    --tokenizer_name $MODEL_DIR/vocab.txt \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --train_file $DATA_DIR/les.tiny.train.json \
    --predict_file $DATA_DIR/les.tiny.dev.json \
    --output_dir models/bert_finetuned_les/ \
    --version_2_with_negative \
    --max_seq_length 512 \
    --max_query_length 30 \
    --max_answer_length 110 \
    --per_gpu_train_batch_size 5 \
    --per_gpu_eval_batch_size 5 \
    --learning_rate 2e-5 \
    --num_train_epochs 2 \
    --doc_stride 128 \
    --logging_steps 10 \
    --save_steps 4000 \
    --eval_steps 4000 \
