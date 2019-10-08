#!/usr/bin/env bash

source_dir="../input/match_featured"
target_dir="../input/ner_featured"

# 重新 split
cat ${source_dir}/split_train_* > ${source_dir}/train_round_0.json
rm ${source_dir}/split_train_*

cat ${source_dir}/split_test_* > ${source_dir}/test_data_r0.json
rm ${source_dir}/split_test_*

# train 和 test 合并，再平均 split
cat ${source_dir}/train_round_0.json ${source_dir}/test_data_r0.json > ${source_dir}/all.json
rm ${source_dir}/train_round_0.json ${source_dir}/test_data_r0.json

# train和test一共 29812，29812 / 4 = 7500
split -d --lines 7500 ${source_dir}/all.json ${source_dir}/split_all_
rm ${source_dir}/all.json

nohup cat ${source_dir}/split_all_00 |python -W ignore 1.3.gen_ner_features.py 0 > ${target_dir}/split_all_ner_00 2>&1 &
nohup cat ${source_dir}/split_all_01 |python -W ignore 1.3.gen_ner_features.py 1 > ${target_dir}/split_all_ner_01 2>&1 &
nohup cat ${source_dir}/split_all_02 |python -W ignore 1.3.gen_ner_features.py 2 > ${target_dir}/split_all_ner_02 2>&1 &
nohup cat ${source_dir}/split_all_03 |python -W ignore 1.3.gen_ner_features.py 3 > ${target_dir}/split_all_ner_03 2>&1 &

## 以下等 NER 抽取完后手动执行：
# cd ../input/ner_featured
# cat split_all_ner_* > all.json
## 切分出 test, 4969 条
# tail -n  4969 all.json > test_data_r0.json
# head -n -4969 all.json > train_round_0.json
# rm all.json
# rm split_all_ner_*
#
## split, 用于 gen_xx_labels 加速
# split -d --lines 1000 train_round_0.json split_train_
# split -d --lines 1000 test_data_r0.json split_test_
