#!/usr/bin/env bash

source_dir="../input/answer_mrc_dataset"
target_dir="../input/answer_mrc_dataset"
max_doc_len=1000
min_ceil_rougel=0.6

nohup cat ${source_dir}/all_back_translate_train_full_content.json |python 4.2.sample_answer_mrc_train_content.py ${max_doc_len} ${min_ceil_rougel} \
        > ${target_dir}/back_translate_train_max_content_len_${max_doc_len}.json 2>&1 &
