#!/usr/bin/env bash

source_dir="../input/cleaned/"
target_dir="../input/mrc_dataset"

nohup cat ${source_dir}/split_back_translated_train_00 |python 3.gen_mrc_dataset.py > ${target_dir}/split_back_translated_train_00 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_01 |python 3.gen_mrc_dataset.py > ${target_dir}/split_back_translated_train_01 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_02 |python 3.gen_mrc_dataset.py > ${target_dir}/split_back_translated_train_02 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_03 |python 3.gen_mrc_dataset.py > ${target_dir}/split_back_translated_train_03 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_04 |python 3.gen_mrc_dataset.py > ${target_dir}/split_back_translated_train_04 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_05 |python 3.gen_mrc_dataset.py > ${target_dir}/split_back_translated_train_05 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_06 |python 3.gen_mrc_dataset.py > ${target_dir}/split_back_translated_train_06 2>&1 &

# 上面程序执行完后，手动执行
# cat ../input/mrc_dataset/split_back_translated_train_0* > ../input/mrc_dataset/all_back_translate_train_full_content.json
# rm ../input/mrc_dataset/split_back_translated_train_0*
