#!/bin/bash
set -ex
DATA_DIR="/home/lq/projects/Research/Reading-Comprehension/les-military-mrc/input"
MODEL_DIR="/home/lq/projects/deep_learning/yingzq/pretrained_weights/chinese_wwm_pytorch"
MODEL_COMMENT="answer_mrc_wwm_BertForLes_ner-kw-pos-wiq_bs32"

python run_les.py \
    --cuda_devices 0,1 \
    --task_name answer_mrc \
    --comment ${MODEL_COMMENT} \
    --model_type bert \
    --customer_model_class BertForLesWithFeatures \
    --model_name_or_path ${MODEL_DIR}/pytorch_model.bin \
    --config_name ${MODEL_DIR}/bert_config.json \
    --tokenizer_name ${MODEL_DIR}/vocab.txt \
    --do_train \
    --do_eval \
    --evaluate_during_training \
    --do_lower_case \
    --train_file ${DATA_DIR}/answer_mrc_dataset/train_max_content_len_1024.json   \
    --predict_file ${DATA_DIR}/dev.json \
    --output_dir answer_models/${MODEL_COMMENT} \
    --version_2_with_negative \
    --max_seq_length 512 \
    --max_query_length 64 \
    --max_answer_length 110 \
    --train_neg_sample_ratio 0.5 \
    --per_gpu_train_batch_size 8 \
    --per_gpu_eval_batch_size 32 \
    --learning_rate 3e-5 \
    --warmup_steps 5200 \
    --warmup_proportion 0.1 \
    --num_train_epochs 6 \
    --gradient_accumulation_steps 2 \
    --doc_stride 400 \
    --logging_steps 400 \
    --save_steps 8000 \
    --eval_steps 4000 \
