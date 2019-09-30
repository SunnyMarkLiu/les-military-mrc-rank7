#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/11 14:48
"""
import sys
sys.path.append('../')
import re
import json
import random
from utils.rouge import RougeL

ans_pattern = re.compile(r'@content\d@')


def find_answer_in_docid(answer):
    docs = ans_pattern.findall(answer)
    return list(set([int(doc[-2:-1]) for doc in docs]))


def cut_doc_where_answer_in(sample, ans_doc_id, ans_doc_id_idxs, max_train_content_len,
                            min_left_context_len, min_right_context_len):
    """
    包含答案的 doc 长度大于 max_doc_len，以答案为基本中心进行切分

    Args:
        ans_doc_id: 答案所在的 doc id
    """
    ans_doc = sample['documents'][ans_doc_id]
    ans_start_idx, ans_end_idx = ans_doc_id_idxs[ans_doc_id]

    # 需要拼接左右剩下的长度
    left_right_len = max_train_content_len - (ans_end_idx - ans_start_idx + 1)  # 减去答案的长度

    left_context_len = random.randint(min_left_context_len, left_right_len - min_right_context_len)
    context_start_idx = ans_start_idx - left_context_len
    context_end_idx = context_start_idx + left_right_len + (ans_end_idx - ans_start_idx + 1)

    new_ans_start_idx = left_context_len
    new_ans_end_idx = new_ans_start_idx + (ans_end_idx - ans_start_idx)

    ans_doc['content'] = ans_doc['content'][context_start_idx: context_end_idx]
    # 特征更新
    # ans_doc[''] = ...
    sample['documents'][ans_doc_id] = ans_doc
    # 更新答案下标
    sample['bridging_entity_labels'] = (ans_doc_id, new_ans_start_idx, new_ans_end_idx)


def sample_train_content(sample, max_train_content_len, min_left_context_len=100, min_right_context_len=50):
    """
    对于全长度的训练集，进行 content 的采样，同时利用滑动窗口，保证 content 长度较小的同时保证足够到的覆盖率

    Args:
        max_train_content_len: 截断的 train content 的最大长度
        min_left_context_len: 答案左侧 context 的最小长度
        min_right_context_len：答案右侧 context 的最小长度
    """
    al = sample['bridging_entity_labels']
    if al:
        answer_in_docs = {al[0]: (al[1], al[2])}
    else:
        answer_in_docs = {}

    for doc_id, doc in enumerate(sample['documents']):
        # 不包含答案的直接截断
        if doc_id not in answer_in_docs:
            doc['content'] = doc['content'][:max_train_content_len]
        else:
            # 包含答案的需要根据答案的位置和 max_train_content_len 的关系进行定位
            start = answer_in_docs[doc_id][0]
            end = answer_in_docs[doc_id][1]

            # 左边 context 的长度稍短，答案从前面截断在前面的 max_train_content_len 内
            if end <= max_train_content_len - min_right_context_len:
                doc['content'] = doc['content'][:max_train_content_len]
            # 右边 context 的长度稍短，答案从后面截断在后面的 max_train_content_len 内
            elif len(doc['content']) - start + min_left_context_len <= max_train_content_len:
                new_ans_start_idx = start - (len(doc['content']) - max_train_content_len)
                new_ans_end_idx = new_ans_start_idx + (end - start)

                doc['content'] = doc['content'][-max_train_content_len:]

                # 更新答案下标
                sample['bridging_entity_labels'] = (doc_id, new_ans_start_idx, new_ans_end_idx)
            # 左边右边的长度都比较长，则以答案为基本中心进行截断
            else:
                cut_doc_where_answer_in(sample, doc_id, answer_in_docs, max_train_content_len,
                                        min_left_context_len, min_right_context_len)

    if sample['bridging_entity'] is not None:
        bridging_entity_labels = sample['bridging_entity_labels']

        if len(bridging_entity_labels) > 0:
            sample['bridging_entity_labels'] = bridging_entity_labels
            sample['fake_bridging_entity'] = sample['documents'][bridging_entity_labels[0]]['content'] \
                [bridging_entity_labels[1]: bridging_entity_labels[2] + 1]
        else:
            sample['bridging_entity_labels'] = []
            sample['fake_bridging_entity'] = ''

        if sample['fake_bridging_entity'] == '':
            sample['ceil_rougel'] = 0
        else:
            ceil_rougel = RougeL().add_inst(cand=sample['fake_bridging_entity'].lower(), ref=sample['bridging_entity'].lower()).get_score()
            sample['ceil_rougel'] = ceil_rougel
    else:
        sample['bridging_entity_labels'] = []
        sample['fake_bridging_entity'] = None
        sample['ceil_rougel'] = -1


if __name__ == '__main__':
    max_train_content_len = int(sys.argv[1])
    min_ceil_rougel = float(sys.argv[2])

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        json_sample = json.loads(line.strip())
        sample_train_content(json_sample, max_train_content_len)

        if json_sample['ceil_rougel'] >= min_ceil_rougel or json_sample['ceil_rougel'] == -1:
            print(json.dumps(json_sample, ensure_ascii=False))
