#!/usr/bin/env bash

echo "convert dataset to dureader format..."
python 0.convert_to_dureader_format.py

echo "split for parallel processing..."
split -d --lines 2500 ../input/raw/train_round_0.json ../input/raw/split_train_
split -d --lines 2500 ../input/raw/test_data_r0.json ../input/raw/split_test_
