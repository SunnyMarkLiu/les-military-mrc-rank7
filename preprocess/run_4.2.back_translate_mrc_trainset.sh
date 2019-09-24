#!/usr/bin/env bash

source_dir="../input/cleaned/back_translate"
target_dir="../input/bridge_entity_mrc_dataset"

# 执行之前，将翻译的训练集放到 cleaned 目录
# split -d --lines 1000 ../input/cleaned/back_translate/back_translated_train_round_0.json ../input/cleaned/back_translate/split_back_translated_train_

nohup cat ${source_dir}/split_back_translated_train_00 |python 2.1.gen_bridge_entity_mrc_dataset.py > ${target_dir}/split_back_translated_train_00 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_01 |python 2.1.gen_bridge_entity_mrc_dataset.py > ${target_dir}/split_back_translated_train_01 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_02 |python 2.1.gen_bridge_entity_mrc_dataset.py > ${target_dir}/split_back_translated_train_02 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_03 |python 2.1.gen_bridge_entity_mrc_dataset.py > ${target_dir}/split_back_translated_train_03 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_04 |python 2.1.gen_bridge_entity_mrc_dataset.py > ${target_dir}/split_back_translated_train_04 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_05 |python 2.1.gen_bridge_entity_mrc_dataset.py > ${target_dir}/split_back_translated_train_05 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_06 |python 2.1.gen_bridge_entity_mrc_dataset.py > ${target_dir}/split_back_translated_train_06 2>&1 &

source_dir="../input/cleaned/back_translate"
target_dir="../input/answer_mrc_dataset"

nohup cat ${source_dir}/split_back_translated_train_00 |python run_2.2.gen_answer_mrc_dataset.sh > ${target_dir}/split_back_translated_train_00 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_01 |python run_2.2.gen_answer_mrc_dataset.sh > ${target_dir}/split_back_translated_train_01 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_02 |python run_2.2.gen_answer_mrc_dataset.sh > ${target_dir}/split_back_translated_train_02 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_03 |python run_2.2.gen_answer_mrc_dataset.sh > ${target_dir}/split_back_translated_train_03 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_04 |python run_2.2.gen_answer_mrc_dataset.sh > ${target_dir}/split_back_translated_train_04 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_05 |python run_2.2.gen_answer_mrc_dataset.sh > ${target_dir}/split_back_translated_train_05 2>&1 &
nohup cat ${source_dir}/split_back_translated_train_06 |python run_2.2.gen_answer_mrc_dataset.sh > ${target_dir}/split_back_translated_train_06 2>&1 &

# 上面程序执行完后，手动执行
# cat ../input/bridge_entity_mrc_dataset/split_back_translated_train_* > ../input/bridge_entity_mrc_dataset/all_back_translate_train_full_content.json
# rm ../input/bridge_entity_mrc_dataset/split_back_translated_train_*

# cat ../input/answer_mrc_dataset/split_back_translated_train_* > ../input/answer_mrc_dataset/all_back_translate_train_full_content.json
# rm ../input/answer_mrc_dataset/split_back_translated_train_*

# 注意对 back translate 生成的训练集进行质量评估（notebook）
