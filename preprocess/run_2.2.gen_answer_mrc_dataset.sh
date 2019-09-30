#!/usr/bin/env bash

source_dir="../input/cleaned/"
target_dir="../input/answer_mrc_dataset/"

echo "generate trainsets..."
nohup cat ${source_dir}split_train_00 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_00 2>&1 &
nohup cat ${source_dir}split_train_01 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_01 2>&1 &
nohup cat ${source_dir}split_train_02 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_02 2>&1 &
nohup cat ${source_dir}split_train_03 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_03 2>&1 &
nohup cat ${source_dir}split_train_04 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_04 2>&1 &
nohup cat ${source_dir}split_train_05 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_05 2>&1 &
nohup cat ${source_dir}split_train_06 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_06 2>&1 &
nohup cat ${source_dir}split_train_07 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_07 2>&1 &
nohup cat ${source_dir}split_train_08 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_08 2>&1 &
nohup cat ${source_dir}split_train_09 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_09 2>&1 &
nohup cat ${source_dir}split_train_10 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_10 2>&1 &
nohup cat ${source_dir}split_train_11 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_11 2>&1 &
nohup cat ${source_dir}split_train_12 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_12 2>&1 &
nohup cat ${source_dir}split_train_13 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_13 2>&1 &
nohup cat ${source_dir}split_train_14 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_14 2>&1 &
nohup cat ${source_dir}split_train_15 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_15 2>&1 &
nohup cat ${source_dir}split_train_16 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_16 2>&1 &
nohup cat ${source_dir}split_train_17 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_17 2>&1 &
nohup cat ${source_dir}split_train_18 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_18 2>&1 &
nohup cat ${source_dir}split_train_19 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_19 2>&1 &
nohup cat ${source_dir}split_train_20 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_20 2>&1 &
nohup cat ${source_dir}split_train_21 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_21 2>&1 &
nohup cat ${source_dir}split_train_22 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_22 2>&1 &
nohup cat ${source_dir}split_train_23 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_23 2>&1 &
nohup cat ${source_dir}split_train_24 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_train_24 2>&1 &

echo "generate testsets..."
nohup cat ${source_dir}split_test_00 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_test_00 2>&1 &
nohup cat ${source_dir}split_test_01 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_test_01 2>&1 &
nohup cat ${source_dir}split_test_02 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_test_02 2>&1 &
nohup cat ${source_dir}split_test_03 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_test_03 2>&1 &
nohup cat ${source_dir}split_test_04 |python 2.2.gen_answer_mrc_dataset.py > ${target_dir}split_test_04 2>&1 &
