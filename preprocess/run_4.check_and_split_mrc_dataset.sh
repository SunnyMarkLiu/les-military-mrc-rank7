#!/usr/bin/env bash

# 注意采样之前需要切分 train/dev，以免扩充的 train 混入 dev

source_dir="../input/mrc_dataset/"      # text_analysis
target_dir="../input/mrc_dataset/"
min_doc_len=600
max_doc_len=800

nohup cat ${source_dir}trainset/train_round_0.json |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}trainset/sample_aug_train_round_0.json 2>&1 &
nohup cat ${source_dir}devset/dev_round_0.json |python 4.check_and_split_mrc_dataset.py ${min_doc_len} ${max_doc_len} > ${target_dir}devset/sample_aug_dev_round_0.json 2>&1 &
