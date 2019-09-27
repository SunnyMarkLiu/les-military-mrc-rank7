#!/usr/bin/env bash

max_doc_len=1000

bridge_entity_mrc_source_dir="../input_rollback_8315/bridge_entity_mrc_dataset"
bridge_entity_mrc_target_dir="../input_rollback_8315/bridge_entity_mrc_dataset"

nohup cat ${bridge_entity_mrc_source_dir}/train.json |python 4.1.sample_bridge_entity_mrc_train_content.py ${max_doc_len} > ${bridge_entity_mrc_target_dir}/train_max_content_len_${max_doc_len}.json 2>&1 &


answer_mrc_source_dir="../input_rollback_8315/answer_mrc_dataset"
answer_mrc_target_dir="../input_rollback_8315/answer_mrc_dataset"

nohup cat ${answer_mrc_source_dir}/train.json |python 4.2.sample_answer_mrc_train_content.py ${max_doc_len} > ${answer_mrc_target_dir}/train_max_content_len_${max_doc_len}.json 2>&1 &
