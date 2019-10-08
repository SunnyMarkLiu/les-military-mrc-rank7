#!/usr/bin/env bash

source_dir="../input/text_featured/"
target_dir="../input/match_featured/"

nohup cat ${source_dir}split_train_00 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_00 2>&1 &
nohup cat ${source_dir}split_train_01 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_01 2>&1 &
nohup cat ${source_dir}split_train_02 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_02 2>&1 &
nohup cat ${source_dir}split_train_03 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_03 2>&1 &
nohup cat ${source_dir}split_train_04 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_04 2>&1 &
nohup cat ${source_dir}split_train_05 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_05 2>&1 &
nohup cat ${source_dir}split_train_06 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_06 2>&1 &
nohup cat ${source_dir}split_train_07 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_07 2>&1 &
nohup cat ${source_dir}split_train_08 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_08 2>&1 &
nohup cat ${source_dir}split_train_09 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_09 2>&1 &
nohup cat ${source_dir}split_train_10 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_10 2>&1 &
nohup cat ${source_dir}split_train_11 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_11 2>&1 &
nohup cat ${source_dir}split_train_12 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_12 2>&1 &
nohup cat ${source_dir}split_train_13 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_13 2>&1 &
nohup cat ${source_dir}split_train_14 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_14 2>&1 &
nohup cat ${source_dir}split_train_15 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_15 2>&1 &
nohup cat ${source_dir}split_train_16 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_16 2>&1 &
nohup cat ${source_dir}split_train_17 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_17 2>&1 &
nohup cat ${source_dir}split_train_18 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_18 2>&1 &
nohup cat ${source_dir}split_train_19 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_19 2>&1 &
nohup cat ${source_dir}split_train_20 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_20 2>&1 &
nohup cat ${source_dir}split_train_21 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_21 2>&1 &
nohup cat ${source_dir}split_train_22 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_22 2>&1 &
nohup cat ${source_dir}split_train_23 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_23 2>&1 &
nohup cat ${source_dir}split_train_24 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_train_24 2>&1 &

nohup cat ${source_dir}split_test_00 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_test_00 2>&1 &
nohup cat ${source_dir}split_test_01 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_test_01 2>&1 &
nohup cat ${source_dir}split_test_02 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_test_02 2>&1 &
nohup cat ${source_dir}split_test_03 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_test_03 2>&1 &
nohup cat ${source_dir}split_test_04 |python -W ignore 1.2.gen_match_features.py > ${target_dir}split_test_04 2>&1 &
