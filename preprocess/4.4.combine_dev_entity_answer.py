#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
bridge entity 和 answer mrc 的 dev 合并

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/10/9 13:12
"""
import json
from tqdm import tqdm

dev_out_f = open('../input/dev.json', 'w')

entity_dev = open('../input/dev_bridge_entity_dense_feat.json')
answer_dev = open('../input/dev_answer.json')
answer_dev = answer_dev.readlines()

dev_out_lines = []
lid = 0
for line in tqdm(entity_dev.readlines()):
    sample = json.loads(line.strip())
    answer_sample = json.loads(answer_dev[lid].strip())

    assert sample['question_id'] == answer_sample['question_id']

    sample['answer_labels'] = answer_sample['answer_labels']
    sample['fake_answers'] = answer_sample['fake_answers']
    sample['answer_ceil_rougel'] = answer_sample['ceil_rougel']
    dev_out_lines.append(json.dumps(sample, ensure_ascii=False) + '\n')
    lid += 1

dev_out_f.writelines(dev_out_lines)
dev_out_f.flush()
dev_out_f.close()
