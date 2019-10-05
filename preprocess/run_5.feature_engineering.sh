#!/usr/bin/env bash

# 执行之前先执行
# split -d --lines 12200 ../input/answer_mrc_dataset/train_max_content_len_1024.json ../input/answer_mrc_dataset/split_train_dataset_
nohup cat ../input/answer_mrc_dataset/split_train_dataset_00 |python 5.1.gen_text_features.py 0 > ../input/answer_mrc_dataset/split_train_dataset_featured_00 2>&1 &
nohup cat ../input/answer_mrc_dataset/split_train_dataset_01 |python 5.1.gen_text_features.py 1 > ../input/answer_mrc_dataset/split_train_dataset_featured_01 2>&1 &
nohup cat ../input/answer_mrc_dataset/dev.json |python 5.1.gen_text_features.py 2 > ../input/answer_mrc_dataset/dev_featured.json 2>&1 &
nohup cat ../input/answer_mrc_dataset/test_r0.json |python 5.1.gen_text_features.py 3 > ../input/answer_mrc_dataset/test_r0_featured.json 2>&1 &
