#!/usr/bin/env bash

source_dir="../input/extracted/"
target_dir="../input/text_analysis/"

echo "cleaning train data..."
nohup cat ${source_dir}split_train_00 ${source_dir}split_train_01 ${source_dir}split_train_02 |python 3.text_analysis.py 0 > ${target_dir}split_train_01 2>&1 &
nohup cat ${source_dir}split_train_03 ${source_dir}split_train_04 ${source_dir}split_train_05 |python 3.text_analysis.py 1 > ${target_dir}split_train_02 2>&1 &
nohup cat ${source_dir}split_train_06 ${source_dir}split_train_07 ${source_dir}split_train_08  ${source_dir}split_train_09 |python 3.text_analysis.py 2 > ${target_dir}split_train_03 2>&1 &

echo "cleaning test data..."
nohup cat ${source_dir}split_test_00 ${source_dir}split_test_01 |python 3.text_analysis.py 3 > ${target_dir}split_test_01 2>&1 &
