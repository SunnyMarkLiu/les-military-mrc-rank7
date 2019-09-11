#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
验证生成的 MRC 的质量，计算 fake answer 与 answer 的 ceil rougel

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/9 18:27
"""
import sys
sys.path.append('../')
import re
import json
import copy
import random
from utils.rouge import RougeL

ans_pattern = re.compile(r'@content\d@')


def find_answer_in_docid(answer):
    docs = ans_pattern.findall(answer)
    return list(set([int(doc[-2:-1]) for doc in docs]))


def calc_ceil_rougel(answer_text, sample):
    # 计算抽取的 fake answer 以及对应的 ceil rougel
    fake_answers = [sample['documents'][answer_label[0]]['content'][answer_label[1]: answer_label[2] + 1]
                    for answer_label in sample['answer_labels']]
    sample['fake_answers'] = fake_answers

    if len(fake_answers) == 0:
        sample['ceil_rougel'] = 0
    else:
        ceil_rougel = RougeL().add_inst(cand=''.join(fake_answers), ref=answer_text).get_score()
        sample['ceil_rougel'] = ceil_rougel


def cut_doc_where_answer_in(sample, ans_doc_id, ans_doc_id_idxs, min_doc_len, max_doc_len):
    """
    包含答案的 doc 长度大于 max_doc_len，以答案为基本中心进行切分

    Args:
        ans_doc_id: 答案所在的 doc id
        ans_doc_id_idxs: dict, {"答案所在的 doc id": [start, end]}
    """
    ans_doc = sample['documents'][ans_doc_id]
    ans_start_idx, ans_end_idx = ans_doc_id_idxs[ans_doc_id]

    # 新doc 的总长度 random(600, 800)
    new_doc_len = random.randint(min_doc_len, max_doc_len)
    # 需要拼接左右剩下的长度
    left_right_len = new_doc_len - (ans_end_idx - ans_start_idx + 1)  # 减去答案的长度

    # 答案左侧内容
    ans_left_content = ans_doc['content'][:ans_start_idx]
    # 答案右侧内容
    ans_right_content = ans_doc['content'][ans_end_idx + 1:]

    if len(ans_left_content) < left_right_len // 2:
        left_len = len(ans_left_content)
    else:
        left_len = random.randint(0, min(len(ans_left_content), left_right_len))

    right_len = left_right_len - left_len

    # 新的答案开始结束下标
    new_ans_start_idx = left_len
    new_ans_end_idx = new_ans_start_idx + (ans_end_idx - ans_start_idx)

    left_context = ans_left_content[-left_len:]
    right_context = ans_right_content[right_len:]
    ans_in_context = ans_doc['content'][ans_start_idx: ans_end_idx + 1]

    context = left_context + ans_in_context + right_context

    ans_doc['content'] = context
    # 特征更新
    # ans_doc[''] = ...
    sample['documents'][ans_doc_id] = ans_doc
    # 更新答案下标
    answer_labels = []
    for al in sample['answer_labels']:
        if ans_doc_id == al[0]:
            answer_labels.append((ans_doc_id, new_ans_start_idx, new_ans_end_idx))
        else:
            answer_labels.append(al)
    sample['answer_labels'] = answer_labels


def window_sample_augment_long_doc_sample(sample, answer_text, ans_doc_id_idxs, min_doc_len, max_doc_len, sample_cnt=4):
    """
    对于 doc 长度大于 max_doc_len 的，以答案位置为基准点，进行
    """
    new_samples = []

    ans_doc_ids = ans_doc_id_idxs.keys()

    for sample_i in range(sample_cnt):
        new_sample = copy.deepcopy(sample)

        new_docs = []
        for doc_id, doc in enumerate(new_sample['documents']):
            if doc_id not in ans_doc_ids:
                left_start = random.randint(0, len(doc['content']) - max_doc_len)
                right_end = random.randint(left_start + min_doc_len, left_start + max_doc_len)

                # 截取 content
                doc['content'] = doc['content'][left_start:right_end]

            new_docs.append(doc)

        new_sample['documents'] = new_docs

        # 对包含答案的 doc 的处理
        for doc_id, doc in enumerate(new_sample['documents']):
            if doc_id in ans_doc_ids:
                cut_doc_where_answer_in(new_sample, doc_id, ans_doc_id_idxs, min_doc_len, max_doc_len)

        # 计算 ceil rouge
        calc_ceil_rougel(answer_text, new_sample)
        new_samples.append(new_sample)
    return new_samples


def check_mrc_dataset(sample, min_doc_len=600, max_doc_len=800):
    """
    验证生成的 MRC 的质量，计算 fake answer 与 answer 的 ceil rougel
    max_doc_len = 800，5个 doc 的长度均超过 800 的有 5238 个，扩充 2 倍则有补充了 10476 个新样本

    - 均大于  max_doc_len=800，5238
    - 部分大于max_doc_len=800，19128
    - 均小于  max_doc_len=800，477

    Args:
        max_doc_len: doc 的最大长度
            答案所在段落超过该长度的，以答案为中心（注意start end），向左右截取采样段落构成新的样本（同时更新start end）；
            不存在答案的超过该长度的，随机选择句子
    """
    answer = sample['answer']
    ans_in_docids = find_answer_in_docid(answer)
    answer_texts = []
    for ans_in_docid in ans_in_docids:
        answer_strs = answer.split('@content{}@'.format(ans_in_docid))
        for answer_str in answer_strs:
            answer_str = answer_str.strip()  # important
            # @content1@ 包裹的实际答案文本
            if answer_str != '' and '@content' not in answer_str:
                answer_str = answer_str.replace('content{}@'.format(ans_in_docid), '')
                answer_texts.append(answer_str)

    # 拼接的答案文本
    answer_text = ''.join(answer_texts)

    # 五篇 doc 长度均小于 max_doc_len
    if sum([len(doc['content']) <= max_doc_len for doc in sample['documents']]) == 5:
        calc_ceil_rougel(answer_text, sample)
        return [sample]

    ans_doc_id_idxs = {x[0]: [x[1], x[2]] for x in sample['answer_labels']}
    # 五篇 doc 长度均大于 max_doc_len，则可进行样本扩充
    if sum([len(doc['content']) > max_doc_len for doc in sample['documents']]) == 5:
        new_samples = window_sample_augment_long_doc_sample(sample, answer_text, ans_doc_id_idxs, min_doc_len, max_doc_len)
        return new_samples

    # 五篇 doc 部分长度大于 max_doc_len
    for doc_id, doc in enumerate(sample['documents']):
        if doc_id in ans_doc_id_idxs:
            # 以答案为中心的裁剪
            cut_doc_where_answer_in(sample, doc_id, ans_doc_id_idxs, min_doc_len, max_doc_len)
        else:
            # 非答案的直接裁剪
            doc['content'] = doc['content'][:max_doc_len]

    calc_ceil_rougel(answer_text, sample)
    return [sample]


if __name__ == '__main__':
    min_doc_len = int(sys.argv[1])
    max_doc_len = int(sys.argv[2])

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        samples = check_mrc_dataset(sample, min_doc_len, max_doc_len)

        for sample in samples:
            print(json.dumps(sample, ensure_ascii=False))
