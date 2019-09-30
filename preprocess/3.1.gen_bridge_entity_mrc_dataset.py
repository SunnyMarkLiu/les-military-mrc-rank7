#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/22 14:41
"""
import sys
sys.path.append('../')
import re
import json
from utils.rouge import RougeL


ans_pattern = re.compile(r'@content\d@')


def find_answer_in_docid(answer):
    docs = ans_pattern.findall(answer)
    return list(set([int(doc[-2:-1]) for doc in docs]))

def find_best_match_support_para(support_text, doc_content):
    """
    利用 support text 长度的窗口滑过 doc_content，计算 rougel 最大的大致位置（粗粒度）
    """
    if support_text in doc_content:
        best_start = doc_content.index(support_text)
        best_end = best_start + len(support_text) - 1
        return best_start, best_end, 1

    if support_text.endswith('。') and support_text[:-1] in doc_content:
        sub_text = support_text[:-1]
        best_start = doc_content.index(sub_text)
        best_end = best_start + len(sub_text) - 1
        return best_start, best_end, 1

        # 存在一些标注错误的样本，去掉空字符后才能定位
    if support_text.replace(' ', '') in doc_content:
        sub_text = support_text.replace(' ', '')
        best_start = doc_content.index(sub_text)
        best_end = best_start + len(sub_text) - 1
        return best_start, best_end, 1

    support_para_chars = set(support_text)
    window_len = len(support_text)     # doc 和 support 不是严格的可定位

    best_score = 0
    best_start = -1
    best_end = -1

    start = 0
    while start < len(doc_content) - window_len - 1:
        while start < len(doc_content) and doc_content[start] not in support_para_chars:
            start += 1

        end = start + window_len
        sub_content = doc_content[start:end + 1]
        score = RougeL().add_inst(cand=sub_content, ref=support_text).get_score()

        if score > best_score:
            best_score = score
            best_start = start
            best_end = end

        start += 1

    if best_score == 0:
        return -1, -1, 0
    else:
        return best_start, best_end, best_score


def find_best_match_answer(answer, support_para):
    """
    找到 sub_text 在 content 覆盖度最大的开始和结束下标（细粒度）
    """
    answer = answer.lower()
    support_para = support_para.lower()
    if answer in support_para:
        best_start = support_para.index(answer)
        best_end = best_start + len(answer) - 1
        return best_start, best_end, 1

    if answer.endswith('。') and answer[:-1] in support_para:
        answer = answer[:-1]
        best_start = support_para.index(answer)
        best_end = best_start + len(answer) - 1
        return best_start, best_end, 1

    # 存在一些标注错误的样本，去掉空字符后才能定位
    if answer.replace(' ', '') in support_para:
        answer = answer.replace(' ', '')
        best_start = support_para.index(answer)
        best_end = best_start + len(answer) - 1
        return best_start, best_end, 1

    # 不能直接定位，利用覆盖率搜索
    support_para_chars = set(answer)

    best_score = 0
    best_start = -1
    best_end = len(support_para) - 1

    for start_idx in range(0, len(support_para) - len(answer)):
        if support_para[start_idx] not in support_para_chars:
            continue

        for end_idx in range(best_end, start_idx - 1, -1):
            if support_para[end_idx] not in support_para_chars:
                continue

            sub_para_content = support_para[start_idx: end_idx + 1]
            score = RougeL().add_inst(cand=sub_para_content, ref=answer).get_score()

            if score > best_score:
                best_score = score
                best_start = start_idx
                best_end = end_idx

    if best_score == 0:
        return -1, -1, 0
    else:
        return best_start, best_end, best_score


def gen_bridging_entity_mrc_dataset(sample):
    """
    生成全文本下的针对 bridging_entity 的 MRC 数据集
    """
    # 段落文本拼接成 content，以及对于的特征的合并
    for doc_id, doc in enumerate(sample['documents']):
        if 'content' in doc: continue
        doc['content'] = ''.join(doc['paragraphs'])
        del doc['paragraphs']

    # 对训练集定位答案的 start end 下标
    if 'bridging_entity' not in sample:
        return

    # 根据 support paragraph 找到答案所在的 sub para
    support_para_in_docids = find_answer_in_docid(sample['supporting_paragraph'])

    supported_paras = {}    # {'support所在doc_id': [{'找到的最匹配的 support para', '最匹配的开始下标', '最匹配的结束下标'}]}
    for sup_para_in_docid in support_para_in_docids:
        para_strs = sample['supporting_paragraph'].split('@content{}@'.format(sup_para_in_docid))
        for para_str in para_strs:
            if para_str != '' and '@content' not in para_str:
                para_str = para_str.replace('content{}@'.format(sup_para_in_docid), '')
                sup_start, sup_end, rougel = find_best_match_support_para(para_str, sample['documents'][sup_para_in_docid - 1]['content'])
                found_sup_para = sample['documents'][sup_para_in_docid - 1]['content'][sup_start: sup_end + 1]
                # 同一个 doc 可能出现多个support para
                if sup_para_in_docid in supported_paras:
                    supported_paras[sup_para_in_docid].append((found_sup_para, sup_start, sup_end))
                else:
                    supported_paras[sup_para_in_docid] = [(found_sup_para, sup_start, sup_end)]

    bridging_entity = sample['bridging_entity']

    # 不存在桥接实体的
    if bridging_entity is None:
        sample['bridging_entity_labels'] = []
        sample['fake_bridging_entity'] = None
        sample['ceil_rougel'] = -1
        return

    max_rougel = 0
    best_start_in_sup_para = -1
    best_end_in_sup_para = -1
    best_sup_doc_i = None
    best_sup_para_i = None

    bridging_entity_labels = []
    for sup_para_in_docid in support_para_in_docids:
        doc_support_paras = supported_paras[sup_para_in_docid]

        for sup_para_i, doc_support_para in enumerate(doc_support_paras):
            start_in_sup_para, end_in_sup_para, rougel = find_best_match_answer(bridging_entity, doc_support_para[0])
            if rougel > max_rougel:
                max_rougel = rougel
                best_start_in_sup_para = start_in_sup_para
                best_end_in_sup_para = end_in_sup_para
                best_sup_doc_i = sup_para_in_docid
                best_sup_para_i = sup_para_i

    if best_start_in_sup_para != -1 and best_end_in_sup_para != -1:
        start_label = best_start_in_sup_para + supported_paras[best_sup_doc_i][best_sup_para_i][1]
        end_label = start_label + (best_end_in_sup_para - best_start_in_sup_para)
        bridging_entity_labels = (best_sup_doc_i - 1, start_label, end_label)

    sample['bridging_entity_labels'] = bridging_entity_labels
    sample['fake_bridging_entity'] = sample['documents'][bridging_entity_labels[0]]['content'] \
                                            [bridging_entity_labels[1]: bridging_entity_labels[2] + 1]

    if sample['fake_bridging_entity'] == '':
        sample['ceil_rougel'] = 0
    else:
        ceil_rougel = RougeL().add_inst(cand=sample['fake_bridging_entity'].lower(), ref=bridging_entity.lower()).get_score()
        sample['ceil_rougel'] = ceil_rougel


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        json_sample = json.loads(line.strip())

        gen_bridging_entity_mrc_dataset(json_sample)
        print(json.dumps(json_sample, ensure_ascii=False))
