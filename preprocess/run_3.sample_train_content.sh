#!/usr/bin/env bash

max_doc_len=1024
min_ceil_rougel=0.2

bridge_entity_mrc_source_dir="../input/bridge_entity_mrc_dataset"
bridge_entity_mrc_target_dir="../input/bridge_entity_mrc_dataset"

nohup cat ${bridge_entity_mrc_source_dir}/train.json |python 3.1.sample_bridge_entity_mrc_train_content.py ${max_doc_len} ${min_ceil_rougel} > ${bridge_entity_mrc_target_dir}/train_max_content_len_${max_doc_len}.json 2>&1 &


answer_mrc_source_dir="../input/answer_mrc_dataset"
answer_mrc_target_dir="../input/answer_mrc_dataset"

nohup cat ${answer_mrc_source_dir}/train.json |python 3.2.sample_answer_mrc_train_content.py ${max_doc_len} ${min_ceil_rougel} > ${answer_mrc_target_dir}/train_max_content_len_${max_doc_len}.json 2>&1 &
