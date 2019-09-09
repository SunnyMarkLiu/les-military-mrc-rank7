#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
计算 document 的 para 与 question 的匹配得分，recall + bleu
按照匹配得分排序，并进行拼接（原始 para 顺序），截取前 1000 个字

为每个段落 i 计算问题与该段落的最大覆盖度，最大覆盖度采用 recall, 再计算该截取片段和问题的 bleu
match_score = recall + bleu

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/5 14:24
"""
import sys

sys.path.append('../')
import sys
import json
import re
from utils.bleu import Bleu


def para_confidence_score(paragraph, question):
    """
    计算问题与该段落的最大匹配得分
    """
    question_chars = set(question)

    best_bleu = 0
    last_end_in_sub_text = len(question) - 1
    for start_idx in range(0, len(paragraph) - len(question)):
        if paragraph[start_idx] not in question_chars:
            continue

        for end_idx in range(last_end_in_sub_text, start_idx - 1, -1):
            if paragraph[end_idx] not in question_chars:
                continue

            sub_para_content = paragraph[start_idx: end_idx + 1]
            rouge = Bleu().add_inst(cand=sub_para_content, ref=question).get_score()

            if rouge > best_bleu:
                best_bleu = rouge
                last_end_in_sub_text = end_idx

    confidence_score = best_bleu
    return confidence_score


ans_pattern = re.compile(r'@content\d@')

def find_answer_in_docid(answer):
    docs = ans_pattern.findall(answer)
    return list(set([int(doc[-2:-1]) for doc in docs]))

def extract_answer_text(answer):
    answer_texts = []
    ans_doc_ids = find_answer_in_docid(answer)
    for ans_doc_id in ans_doc_ids:
        answer_strs = answer.split('@content{}@'.format(ans_doc_id))
        for answer_str in answer_strs:
            answer_str = answer_str.strip()  # important
            # @content1@ 包裹的实际答案文本
            if answer_str != '' and '@content' not in answer_str:
                answer_str = answer_str.replace('content{}@'.format(ans_doc_id), '')
                answer_texts.append(answer_str)

    return ''.join(answer_texts)


def extract_paragraph(sample, max_doc_len):
    """
    段落排序和抽取
    """
    if 'supporting_paragraph' in sample:
        ref_text = extract_answer_text(sample['supporting_paragraph'])
    else:
        ref_text = sample['question'] + sample['keyword']

    for doc_id, doc in enumerate(sample['documents']):

        # 计算每个doc的 paragraph 的置信度得分
        para_infos = []
        doc_len = 0
        for para_id, para in enumerate(doc['paragraphs']):
            confidence_score = para_confidence_score(para, ref_text)
            # ((段落匹配得分，段落长度)，段落的原始下标)
            para_infos.append((confidence_score, len(para), para_id))
            doc_len += len(para)

        # 对于doc总长度小于 max_doc_len 的不就行段落筛选
        if doc_len < max_doc_len:
            continue

        # 按照 match_score 降序排列，按照段落长度升序排列，以保证较短的置信度高的段落能召回
        para_infos.sort(key=lambda x: (-x[0], x[1]))

        # 依据 max_doc_len 筛选段落
        selected_para_ids = []
        last_para_id = -1
        last_para_cut_char_idx = -1  # 如果超出范围，最后一个段落需要截断的 char id
        selected_char_len = 0  # 拼接的 char 总长度

        for para_info in para_infos:
            para_id, para_len = para_info[2], para_info[1]
            selected_char_len += para_len
            if selected_char_len <= max_doc_len:
                selected_para_ids.append(para_id)
            else:  # 超出 max_doc_len 范围，截取到 max_doc_len 长度，防止筛掉了答案所在的段落
                last_para_id = para_id
                last_para_cut_char_idx = max_doc_len - selected_char_len + 1
                break

        # 按照 para id 的原始顺序排列
        selected_para_ids.sort()

        # 拼接筛选的段落
        new_paragraphs = [doc['paragraphs'][para_id] for para_id in selected_para_ids]

        if last_para_id != -1:
            last_para_content = doc['paragraphs'][last_para_id][:last_para_cut_char_idx]
            new_paragraphs.append(last_para_content)

        doc['paragraphs'] = new_paragraphs

if __name__ == '__main__':
    max_doc_len = int(sys.argv[1])

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        extract_paragraph(sample, max_doc_len)
        print(json.dumps(sample, ensure_ascii=False))
