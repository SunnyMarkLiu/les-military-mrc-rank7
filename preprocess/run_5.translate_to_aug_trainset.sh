#!/usr/bin/env bash

source_dir="../input/mrc_dataset"
target_dir="../input/mrc_dataset"

nohup cat ${source_dir}/train.json |python 5.translate_to_aug_trainset.py > ${target_dir}/back_translate_aug_train.json 2>&1 &
