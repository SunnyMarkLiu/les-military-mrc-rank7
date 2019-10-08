#!/usr/bin/env bash

max_doc_len=1024
min_ceil_rougel=0.2

bridge_entity_mrc_source_dir="../input/bridge_entity_labels"
bridge_entity_mrc_target_dir="../input/bridge_entity_labels"

nohup cat ${bridge_entity_mrc_source_dir}/train_full_content.json |python 4.1.sample_bridge_entity_mrc_train_content.py ${max_doc_len} ${min_ceil_rougel} > ${bridge_entity_mrc_target_dir}/train_max_content_len_${max_doc_len}.json 2>&1 &

answer_mrc_source_dir="../input/answer_labels"
answer_mrc_target_dir="../input/answer_labels"

nohup cat ${answer_mrc_source_dir}/train_full_content.json |python 4.2.sample_answer_mrc_train_content.py ${max_doc_len} ${min_ceil_rougel} > ${answer_mrc_target_dir}/train_max_content_len_${max_doc_len}.json 2>&1 &
