#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
根据 answer 长度分布，进行 bin 切分，从 bin 中进行采样得到 dev

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/10/6 21:11
"""
import sys
import json

with open('2475_dev_sample_ques_ids.txt', 'r') as f:
    dev_ques_ids = f.readline().strip().split(',')

dev_ques_ids = set(dev_ques_ids)

if __name__ == '__main__':
    data_type = sys.argv[1]

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())

        if data_type == 'dev' and sample['question_id'] in dev_ques_ids:
            print(json.dumps(sample, ensure_ascii=False))

        if data_type == 'train' and sample['question_id'] not in dev_ques_ids:
            print(json.dumps(sample, ensure_ascii=False))
