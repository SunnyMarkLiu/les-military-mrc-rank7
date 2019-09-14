#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
莱斯杯 2019 年数据

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/13 20:49
"""
import json
from tqdm import tqdm

military_corpus_fout = open('../../input/military_corpus/les_2019.txt', 'w', encoding='utf8')

for path in ['all_train_full_content.json', 'test_r0.json']:
    with open('../../input/mrc_dataset/' + path) as f:
        lines = f.readlines()
        for line in tqdm(lines):
            sample = json.loads(line)

            for doc in sample['documents']:
                content = doc['content']
                sents = [sent + '。\n' for sent in content.split('。') if len(sent) > 3]
                sents.append('\n')
                military_corpus_fout.writelines(sents)

military_corpus_fout.flush()
