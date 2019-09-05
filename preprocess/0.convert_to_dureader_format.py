#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
将数据转换为 dureader 格式

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/3 16:21
"""
import re
import json
import pandas as pd
from tqdm import tqdm
tqdm.pandas()

ans_pattern = re.compile(r'@content\d@')


def find_answer_in_docid(answer):
    docs = ans_pattern.findall(answer)
    return list(set([int(doc[-2:-1]) for doc in docs]))


train_df = pd.read_csv('../input/original/train_round_0.csv', sep=',')
test_df = pd.read_csv('../input/original/test_data_r0.csv', sep=',')

train_file_out = open('../input/raw/train_round_0.json', 'w', encoding='utf8')
test_file_out = open('../input/raw/test_data_r0.json', 'w', encoding='utf8')

train_samples = []
for rid, row in tqdm(train_df.iterrows(), total=train_df.shape[0]):
    sample = {'question_id': row['question_id'],
              'question': row['question'],
              'answer': row['answer'],
              'bridging_entity': None if row['bridging_entity'] == '无' else row['bridging_entity'],
              'keyword': row['keyword'],
              'supporting_paragraph': row['supporting_paragraph'],
              'documents': []}

    ans_in_doc_ids = find_answer_in_docid(row['answer'])
    supported_doc_ids = find_answer_in_docid(row['supporting_paragraph'])

    for docid in range(1, 6):
        if docid in ans_in_doc_ids:
            is_selected = True
        else:
            is_selected = False

        # 注意：
        # 1. supporting_paragraph存在句号，所以在转成dureader时候按照句号进行切分句子存在缺陷！
        # 2. 观察数据发现按照双空格'  '划分段落
        paragraphs = [para.strip() for para in row['content{}'.format(docid)].split('  ')]
        paragraphs = [para for para in paragraphs if para != '']

        supported_para_ids = []

        sample['documents'].append({
            'is_selected': is_selected,
            'supported_para_ids': supported_para_ids,
            'title': row['title{}'.format(docid)],
            'paragraphs': paragraphs
        })

    train_samples.append(json.dumps(sample, ensure_ascii=False) + '\n')

    if len(train_samples) % 1000 == 0:
        train_file_out.writelines(train_samples)
        train_file_out.flush()
        train_samples.clear()

if train_samples:
    train_file_out.writelines(train_samples)
    train_file_out.flush()
    train_file_out.close()

# --------------------------------- test ---------------------------------
test_samples = []
for rid, row in tqdm(test_df.iterrows(), total=test_df.shape[0]):
    sample = {'question_id': row['question_id'],
              'question': row['question'],
              'keyword': row['keyword'],
              'documents': []}

    for docid in range(1, 6):
        # 注意：
        # 1. supporting_paragraph存在句号，所以在转成dureader时候按照句号进行切分句子存在缺陷！
        # 2. 观察数据发现按照双空格'  '划分段落
        paragraphs = [para.strip() for para in row['content{}'.format(docid)].split('  ')]
        paragraphs = [para for para in paragraphs if para != '']

        sample['documents'].append({
            'title': row['title{}'.format(docid)],
            'paragraphs': paragraphs
        })

    test_samples.append(json.dumps(sample, ensure_ascii=False) + '\n')

    if len(test_samples) % 1000 == 0:
        test_file_out.writelines(test_samples)
        test_file_out.flush()
        test_samples.clear()

if test_samples:
    test_file_out.writelines(test_samples)
    test_file_out.flush()
    test_file_out.close()
