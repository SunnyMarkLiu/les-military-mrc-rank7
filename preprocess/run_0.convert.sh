#!/usr/bin/env bash

echo "convert dataset to dureader format..."
python 0.convert_to_dureader_format.py

echo "split for parallel processing..."
split -d --lines 1000 ../input_new/raw/train.json ../input_new/raw/split_train_
split -d --lines 1000 ../input_new/raw/test_data_r0.json ../input_new/raw/split_test_
