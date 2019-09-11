#!/usr/bin/env bash

source_dir="../input/mrc_dataset"
target_dir="../input/mrc_dataset"
max_doc_len=1000

nohup cat ${source_dir}/train.json |python 4.sample_train_content.py ${max_doc_len} > ${target_dir}/train_max_content_len_${max_doc_len}.json 2>&1 &
