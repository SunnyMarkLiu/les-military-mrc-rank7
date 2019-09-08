#!/usr/bin/env bash

source_dir="../input/text_analysis/"
target_dir="../input/mrc_dataset/"

nohup cat ${source_dir}split_train_01 |python 5.gen_mrc_label.py > ${target_dir}split_train_01 2>&1 &
nohup cat ${source_dir}split_train_02 |python 5.gen_mrc_label.py > ${target_dir}split_train_02 2>&1 &
nohup cat ${source_dir}split_train_03 |python 5.gen_mrc_label.py > ${target_dir}split_train_03 2>&1 &
nohup cat ${source_dir}split_test_01 |python 5.gen_mrc_label.py > ${target_dir}split_test_01 2>&1 &
