#!/usr/bin/env bash

# 划分训练集和验证集
target_dir="../input/mrc_dataset"

cat ${target_dir}split_train_* > ${target_dir}/all_train_full_content.json

tail -n  500 ${target_dir}/all_train_full_content.json > ${target_dir}/dev.json
head -n -500 ${target_dir}/all_train_full_content.json > ${target_dir}/train.json

wc -l ${target_dir}/*
