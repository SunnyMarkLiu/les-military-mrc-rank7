#!/usr/bin/env bash

data_dir="../input/cleaned"

# 重新 split
cat ${data_dir}/split_train_* > ${data_dir}/train_round_0.json
rm ${data_dir}/split_train_*

cat ${data_dir}/split_test_* > ${data_dir}/test_data_r0.json
rm ${data_dir}/split_test_*

# train 和 test 合并，再平均 split
cat ${data_dir}/train_round_0.json ${data_dir}/test_data_r0.json > ${data_dir}/all.json
rm ${data_dir}/train_round_0.json ${data_dir}/test_data_r0.json
# train和test一共 29812，29812 / 4 = 7500
split -d --lines 7500 ${data_dir}/all.json ${data_dir}/split_all_
rm ${data_dir}/all.json

nohup cat ${data_dir}/split_all_00 |python 2.3.gen_ner_features.py 0 > ${data_dir}/split_all_ner_00 2>&1 &
nohup cat ${data_dir}/split_all_01 |python 2.3.gen_ner_features.py 1 > ${data_dir}/split_all_ner_01 2>&1 &
nohup cat ${data_dir}/split_all_02 |python 2.3.gen_ner_features.py 2 > ${data_dir}/split_all_ner_02 2>&1 &
nohup cat ${data_dir}/split_all_03 |python 2.3.gen_ner_features.py 3 > ${data_dir}/split_all_ner_03 2>&1 &

## 以下等 NER 抽取完后手动执行：
#cat ${data_dir}/split_all_ner_* > ${data_dir}/all.json
## 切分出 test, 4969 条
#tail -n  4969 ${data_dir}/all.json > ${data_dir}/test_data_r0.json
#head -n -4969 ${data_dir}/all.json > ${data_dir}/train_round_0.json
#rm ${data_dir}/all.json
#
## split, 用于 gen_xx_labels 加速
#split -d --lines 1000 ${data_dir}/train_round_0.json ${data_dir}/split_train_
#split -d --lines 1000 ${data_dir}/test_data_r0.json ${data_dir}/split_test_
#rm ${data_dir}/*.json
