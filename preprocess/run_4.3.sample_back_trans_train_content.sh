#!/usr/bin/env bash

max_doc_len=1000
min_ceil_rougel=0.6

source_dir="../input/bridge_entity_mrc_dataset"
target_dir="../input/bridge_entity_mrc_dataset"

nohup cat ${source_dir}/all_back_translate_train_full_content.json |python 3.1.sample_bridge_entity_mrc_train_content.py ${max_doc_len} ${min_ceil_rougel} > ${target_dir}/back_translate_train_max_content_len_${max_doc_len}.json 2>&1 &

source_dir="../input/answer_mrc_dataset"
target_dir="../input/answer_mrc_dataset"

nohup cat ${source_dir}/all_back_translate_train_full_content.json |python 3.2.sample_answer_mrc_train_content.py ${max_doc_len} ${min_ceil_rougel} > ${target_dir}/back_translate_train_max_content_len_${max_doc_len}.json 2>&1 &
