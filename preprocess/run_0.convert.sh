#!/usr/bin/env bash

echo "convert dataset to dureader format..."
python 0.convert_to_dureader_format.py

echo "generate dev question ids"
python 0.1.ans_len_bin_sample_dev.py

echo "split for parallel processing..."
split -d --lines 1000 ../input/raw/train_round_0.json ../input/raw/split_train_
split -d --lines 1000 ../input/raw/test_data_r0.json ../input/raw/split_test_
