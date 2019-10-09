#!/usr/bin/env bash

echo "dev/test 特征压缩"
cat ../input/ner_featured/test_data_r0.json |python 4.3.dense_dev_test_feature_list.py > ../input/test_data_r0.json
cat ../input/dev_bridge_entity.json |python 4.3.dense_dev_test_feature_list.py > ../input/dev_bridge_entity_dense_feat.json

echo "bridge entity 和 answer mrc 的 dev 合并"
python 4.4.combine_dev_entity_answer.py

rm ../input/dev_answer.json
rm ../input/dev_bridge_entity_dense_feat.json
