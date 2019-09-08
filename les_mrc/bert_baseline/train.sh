#!/bin/bash
set -ex
DATA_DIR="/home/lq/projects/Research/Reading-Comprehension/les-military-mrc/input/mrc_dataset"
MODEL_DIR="/home/lq/projects/deep_learning/yingzq/pretrained_weights/chinese_wwm_ext_pytorch"

python run_les.py \
    --cuda_devices 1,2,3 \
    --model_type bert \
    --model_name_or_path $MODEL_DIR/pytorch_model.bin \
    --config_name $MODEL_DIR/bert_config.json \
    --tokenizer_name $MODEL_DIR/vocab.txt \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --train_file $DATA_DIR/les.train.json \
    --predict_file $DATA_DIR/les.dev.json \
    --output_dir models/bert_finetuned_les_wwm_ext/ \
    --version_2_with_negative \
    --max_seq_length 512 \
    --max_query_length 64 \
    --max_answer_length 110 \
    --per_gpu_train_batch_size 7 \
    --per_gpu_eval_batch_size 24 \
    --learning_rate 5e-5 \
    --warmup_steps 0 \
    --num_train_epochs 4 \
    --doc_stride 128 \
    --logging_steps 500 \
    --save_steps 5000 \
    --eval_steps 4000 \
