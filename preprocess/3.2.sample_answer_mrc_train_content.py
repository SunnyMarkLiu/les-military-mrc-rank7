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


def calc_ceil_rougel(answer_text, sample):
    # 计算抽取的 fake answer 以及对应的 ceil rougel
    fake_answers = [sample['documents'][answer_label[0]]['content'][answer_label[1]: answer_label[2] + 1]
                    for answer_label in sample['answer_labels']]
    sample['fake_answers'] = fake_answers

    if len(fake_answers) == 0:
        sample['ceil_rougel'] = 0
    else:
        ceil_rougel = RougeL().add_inst(cand=''.join(fake_answers).lower(), ref=answer_text.lower()).get_score()
        sample['ceil_rougel'] = ceil_rougel


def cut_doc_where_answer_in(sample, ans_doc_id, ans_start_idx, ans_end_idx, max_train_content_len,
                            min_left_context_len, min_right_context_len):
    """
    包含答案的 doc 长度大于 max_doc_len，以答案为基本中心进行切分

    Args:
        ans_doc_id: 答案所在的 doc id
    """
    ans_doc = sample['documents'][ans_doc_id]

    # 需要拼接左右剩下的长度
    left_right_len = max_train_content_len - (ans_end_idx - ans_start_idx + 1)  # 减去答案的长度

    left_context_len = random.randint(min_left_context_len, left_right_len - min_right_context_len)
    context_start_idx = ans_start_idx - left_context_len
    context_end_idx = context_start_idx + left_right_len + (ans_end_idx - ans_start_idx + 1)

    new_ans_start_idx = left_context_len
    new_ans_end_idx = new_ans_start_idx + (ans_end_idx - ans_start_idx)

    ans_doc['content'] = ans_doc['content'][context_start_idx: context_end_idx]
    ans_doc['supported_para_mask'] = ans_doc['supported_para_mask'][context_start_idx: context_end_idx]
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
    return context_start_idx


def sample_train_content(sample, max_train_content_len, min_left_context_len=50, min_right_context_len=50):
    """
    对于全长度的训练集，进行 content 的采样，同时利用滑动窗口，保证 content 长度较小的同时保证足够到的覆盖率

    Args:
        max_train_content_len: 截断的 train content 的最大长度
        min_left_context_len: 答案左侧 context 的最小长度
        min_right_context_len：答案右侧 context 的最小长度
    """
    answer_in_docs = {}

    # 注意一个 doc 中可能存在多个答案
    for al in sample['answer_labels']:
        if al[0] in answer_in_docs:
            answer_in_docs[al[0]].append((al[1], al[2]))
        else:
            answer_in_docs[al[0]] = [(al[1], al[2])]

    for doc_id, doc in enumerate(sample['documents']):
        if len(doc['content']) < max_train_content_len:
            continue

        # 不包含答案的直接截断
        if doc_id not in answer_in_docs:
            doc['content'] = doc['content'][:max_train_content_len]
        else:
            # doc 中只包含一个答案
            if len(answer_in_docs[doc_id]) == 1:
                # 包含答案的需要根据答案的位置和 max_train_content_len 的关系进行定位
                start = answer_in_docs[doc_id][0][0]
                end = answer_in_docs[doc_id][0][1]

                # 左边 context 的长度稍短，答案从前面截断在前面的 max_train_content_len 内
                if end <= max_train_content_len - min_right_context_len:
                    doc['content'] = doc['content'][:max_train_content_len]
                    doc['supported_para_mask'] = doc['supported_para_mask'][:max_train_content_len]
                # 右边 context 的长度稍短，答案从后面截断在后面的 max_train_content_len 内
                elif len(doc['content']) - start + min_left_context_len <= max_train_content_len:
                    new_ans_start_idx = start - (len(doc['content']) - max_train_content_len)
                    new_ans_end_idx = new_ans_start_idx + (end - start)

                    doc['content'] = doc['content'][-max_train_content_len:]
                    doc['supported_para_mask'] = doc['supported_para_mask'][-max_train_content_len:]

                    # 更新答案下标
                    answer_labels = []
                    for al in sample['answer_labels']:
                        if doc_id == al[0]:
                            answer_labels.append((doc_id, new_ans_start_idx, new_ans_end_idx))
                        else:
                            answer_labels.append(al)
                    sample['answer_labels'] = answer_labels
                # 左边右边的长度都比较长，则以答案为基本中心进行截断
                else:
                    cut_doc_where_answer_in(sample, doc_id, start, end, max_train_content_len,
                                            min_left_context_len, min_right_context_len)
            else:
                # 同一个 doc 中存在多个答案的情况
                starts = [al[0] for al in answer_in_docs[doc_id]]
                ends   = [al[1] for al in answer_in_docs[doc_id]]
                min_start = min(starts)
                max_end = max(ends)

                # 左边 context 的长度稍短，答案从前面截断在前面的 max_train_content_len 内
                if max_end < max_train_content_len - min_right_context_len:
                    doc['content'] = doc['content'][:max_train_content_len]
                    doc['supported_para_mask'] = doc['supported_para_mask'][:max_train_content_len]
                # 右边 context 的长度稍短，答案从后面截断在后面的 max_train_content_len 内
                elif len(doc['content']) - min_start + min_left_context_len <= max_train_content_len:
                    # 更新多答案的下标
                    cur_doc_answers = []
                    for start, end in zip(starts, ends):
                        new_start = start - (len(doc['content']) - max_train_content_len)
                        new_end = new_start + (end - start)
                        cur_doc_answers.append((doc_id, new_start, new_end))

                    doc['content'] = doc['content'][-max_train_content_len:]
                    doc['supported_para_mask'] = doc['supported_para_mask'][-max_train_content_len:]

                    answer_labels = []
                    for al in sample['answer_labels']:
                        if doc_id != al[0]:
                            answer_labels.append(al)
                    answer_labels.extend(cur_doc_answers)
                    sample['answer_labels'] = answer_labels

                # 左边右边的长度都比较长，则以多答案为基本中心进行截断
                else:
                    if max_end - min_start < max_train_content_len - min_left_context_len - min_right_context_len:
                        # 多答案看成一个答案为中心进行切分
                        doc_start_idx = cut_doc_where_answer_in(
                            sample, doc_id, min_start, max_end, max_train_content_len,
                            min_left_context_len, min_right_context_len
                        )
                        # 更新多答案的下标
                        cur_doc_answers = []
                        for start, end in zip(starts, ends):
                            new_start = start - doc_start_idx
                            new_end = end - doc_start_idx
                            cur_doc_answers.append((doc_id, new_start, new_end))

                        answer_labels = []
                        for al in sample['answer_labels']:
                            if doc_id != al[0]:
                                answer_labels.append(al)
                        answer_labels.extend(cur_doc_answers)
                        sample['answer_labels'] = answer_labels
                    # TODO:min start 和 max end 跨度太长，暂时直接丢弃随机丢弃中间的文本
                    # else:
                    #     gap = max_end - min_start + 1
                    #     max_train_content_len -

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
    calc_ceil_rougel(answer_text, sample)


if __name__ == '__main__':
    max_train_content_len = int(sys.argv[1])
    min_ceil_rougel = float(sys.argv[2])

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        json_sample = json.loads(line.strip())
        sample_train_content(json_sample, max_train_content_len)

        if json_sample['ceil_rougel'] >= min_ceil_rougel:
            print(json.dumps(json_sample, ensure_ascii=False))
