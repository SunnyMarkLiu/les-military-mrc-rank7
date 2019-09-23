#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
将预测的 bridge entity 和 question 拼接，构成新的测试集

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/22 19:09
"""
import json

with open('models/bert_wwm_BertForLes_bridge/predictions_les_dev.json') as f:
    test_pred_bridge_entity = json.load(f)

with open('../../input/answer_mrc_dataset_test/dev_combine_pred_bridge_entity.json', 'w', encoding='utf8') as fout:
    with open('../../input/answer_mrc_dataset_test/dev.json') as f:
        for line in f:
            sample = json.loads(line.strip())
            sample['question'] = sample['question'] + test_pred_bridge_entity[sample['question_id']]
            fout.write(json.dumps(sample, ensure_ascii=False) + '\n')


# with open('models/bert_wwm_BertForLes_test/predictions_checkpoint_dev.json') as f:
#     test_pred_bridge_entity = json.load(f)
#
# with open('../../input/answer_mrc_dataset/test_r0_combine_pred_bridge_entity.json', 'w', encoding='utf8') as fout:
#     with open('../../input/answer_mrc_dataset/test_r0.json') as f:
#         for line in f:
#             sample = json.loads(line.strip())
#             sample['question'] = test_pred_bridge_entity[sample['question_id']] + sample['question']
#             fout.write(json.dumps(sample, ensure_ascii=False) + '\n')
