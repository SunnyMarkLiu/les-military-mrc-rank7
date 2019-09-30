#!/usr/bin/env bash

# clean 目录先执行 split -d --lines 500 back_translate_aug_cleaned_train.json split_back_translated_train_

source_dir="../input/cleaned/"
target_dir="../input/answer_mrc_dataset"

nohup cat ${source_dir}/split_back_translated_train_00 |python 3.2.gen_answer_mrc_dataset.py > ${target_dir}/split_back_translated_train_00 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_01 |python 3.2.gen_answer_mrc_dataset.py > ${target_dir}/split_back_translated_train_01 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_02 |python 3.2.gen_answer_mrc_dataset.py > ${target_dir}/split_back_translated_train_02 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_03 |python 3.2.gen_answer_mrc_dataset.py > ${target_dir}/split_back_translated_train_03 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_04 |python 3.2.gen_answer_mrc_dataset.py > ${target_dir}/split_back_translated_train_04 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_05 |python 3.2.gen_answer_mrc_dataset.py > ${target_dir}/split_back_translated_train_05 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_06 |python 3.2.gen_answer_mrc_dataset.py > ${target_dir}/split_back_translated_train_06 2>&1 &

# 上面程序执行完后，手动执行
# cat ../input/answer_mrc_dataset/split_back_translated_train_0* > ../input/answer_mrc_dataset/all_back_translate_train_full_content.json
# rm ../input/answer_mrc_dataset/split_back_translated_train_0*
