#!/usr/bin/env bash

echo "convert dataset to dureader format..."
python 0.convert_to_dureader_format.py

echo "cleaning train data..."
cat ../input/raw/train_round_0.json |python 1.text_cleaning.py > ../input/cleaned/train_round_0.json
echo "cleaning test data..."
cat ../input/raw/test_data_r0.json |python 1.text_cleaning.py > ../input/cleaned/test_data_r0.json

echo "remove not related paras for train data ..."
cat ../input/cleaned/train_round_0.json |python 2.remove_not_related_paras.py recall 0.03 > ../input/extracted/train_round_0.json
echo "remove not related paras for test data ..."
cat ../input/cleaned/test_data_r0.json |python 2.remove_not_related_paras.py recall 0.03 > ../input/extracted/test_data_r0.json
