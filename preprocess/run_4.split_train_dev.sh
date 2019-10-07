#!/usr/bin/env bash

# 划分训练集和验证集
target_dir="../input/bridge_entity_labels"

cat ${target_dir}/split_train_* > ${target_dir}/all_train_full_content.json
rm ${target_dir}/split_*

cat ${target_dir}/all_train_full_content.json |python 2.3.split_train_dev.py train > ${target_dir}/train_full_content.json
cat ${target_dir}/all_train_full_content.json |python 2.3.split_train_dev.py dev > ${target_dir}/dev.json

wc -l ${target_dir}/*

# 划分训练集和验证集
target_dir="../input/answer_labels"

cat ${target_dir}/split_train_* > ${target_dir}/all_train_full_content.json
rm ${target_dir}/split_*

cat ${target_dir}/all_train_full_content.json |python 2.3.split_train_dev.py train > ${target_dir}/train_full_content.json
cat ${target_dir}/all_train_full_content.json |python 2.3.split_train_dev.py dev > ${target_dir}/dev.json

wc -l ${target_dir}/*
