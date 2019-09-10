#!/usr/bin/env bash

source_dir="../input/mrc_dataset/"      # text_analysis
target_dir="../input/aug_mrc_dataset/"
min_doc_len=600
max_doc_len=800

nohup cat ${source_dir}split_train_00 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_00 2>&1 &
nohup cat ${source_dir}split_train_01 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_01 2>&1 &
nohup cat ${source_dir}split_train_02 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_02 2>&1 &
nohup cat ${source_dir}split_train_03 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_03 2>&1 &
nohup cat ${source_dir}split_train_04 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_04 2>&1 &
nohup cat ${source_dir}split_train_05 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_05 2>&1 &
nohup cat ${source_dir}split_train_06 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_06 2>&1 &
nohup cat ${source_dir}split_train_07 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_07 2>&1 &
nohup cat ${source_dir}split_train_08 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_08 2>&1 &
nohup cat ${source_dir}split_train_09 |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}split_train_09 2>&1 &
