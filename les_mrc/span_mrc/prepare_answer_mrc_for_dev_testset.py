#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
将 bridge entity mrc 模型预测的 entity 添加到 bridging_entity 字段上，用于 answer mrc 的预测

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/24 18:24
"""
import json

# ----------- test -----------
with open('bridge_models/bridge_mrc_wwm_BertForLes/checkpoint-best/predictions_checkpoint_test.json') as f:
    test_bridge_entity = json.load(f)

with open('../../input/answer_mrc_dataset/add_pred_bridging_entity_test_r0.json', 'w', encoding='utf8') as fout:
    with open('../../input/answer_mrc_dataset/test_r0.json') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            sample['bridging_entity'] = test_bridge_entity[sample['question_id']]
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')

# ----------- dev -----------
with open('bridge_models/bridge_mrc_wwm_BertForLes/checkpoint-best/predictions_checkpoint_dev.json') as f:
    dev_bridge_entity = json.load(f)

with open('../../input/answer_mrc_dataset/add_pred_bridging_entity_dev.json', 'w', encoding='utf8') as fout:
    with open('../../input/answer_mrc_dataset/dev.json') as fin:
        for line in fin:
            sample = json.loads(line.strip())
            sample['bridging_entity'] = dev_bridge_entity[sample['question_id']]
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
