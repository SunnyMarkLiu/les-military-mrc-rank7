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
from utils.bleu import Bleu
from utils.rouge import RougeL
from utils.precision_recall_f1 import precision_recall_f1


def current_para_cross_paras_rougel(para_id, paras):
    """
    计算该 para 与该 doc 中其他 para 的 rougel 得分均值，用于评判该段落和其他段落的交互（反映该段落的主题程度）
    """
    if len(paras) == 1:     # 只有一个段落，则保留且 cross_paras_rougel = 1
        return 1.0

    cur_para = paras[para_id]

    rouge_ls = []
    for pid, para in enumerate(paras):
        if pid == para_id:
            continue

        # 计算两个段落间的 rouge 得分
        rougel = RougeL().add_inst(cand=para, ref=cur_para).get_score()
        rouge_ls.append(rougel)

    return sum(rouge_ls) / len(rouge_ls)


def para_question_best_coverage_score(paragraph, question):
    """
    计算 para 和 question 的匹配得分（最大覆盖率）
    为每个段落 i 计算问题与该段落的最大覆盖度，最大覆盖度采用 recall, 再计算该截取片段和问题的 bleu
    match_score = recall + bleu
    """
    paragraph = list(paragraph)
    question = list(question)

    best_recall = 0
    best_start_idx = -1
    best_end_idx = -1
    for start_idx in range(0, len(paragraph)):
        # 开始的词不在答案中，或者，开始的词为标点符号或splitter，直接过滤
        if paragraph[start_idx] not in question:
            continue

        for end_idx in range(len(paragraph) - 1, start_idx - 1, -1):
            if paragraph[end_idx] not in question:
                continue

            sub_para_content = paragraph[start_idx: end_idx + 1]
            # 计算该片段和 question 的 recall
            recall = precision_recall_f1(sub_para_content, question)[1]

            if recall > best_recall:
                best_recall = recall
                best_start_idx = start_idx
                best_end_idx = end_idx

    best_sub_para = paragraph[best_start_idx: best_end_idx + 1]
    best_sub_para = ''.join(best_sub_para)
    question = ''.join(question)
    # 计算匹配度最高情况下的 bleu4
    if best_sub_para == '':
        bleu4 = 0
    else:
        bleu4 = Bleu(4).add_inst(cand=best_sub_para, ref=question).get_score()
    best_coverage_score = best_recall + bleu4
    return best_coverage_score


def para_confidence_score(para_id, paras, question):
    """
    计算该段落的置信度得分，用于排序筛选段落
    """
    paragraph = paras[para_id]

    # 为每个段落 i 计算问题与该段落的最大覆盖度, recall + bleu
    best_coverage_score = para_question_best_coverage_score(paragraph, question)
    # 计算该 para 与该 doc 中其他 para 的 rougel 得分均值，用于评判该段落和其他段落的交互（反映该段落的主题程度）
    cross_paras_rougel = current_para_cross_paras_rougel(para_id, paras)

    confidence_score = best_coverage_score + cross_paras_rougel
    return confidence_score


def extract_paragraph(sample, max_doc_len):
    """
    段落排序和抽取
    """
    question = sample['question']
    for doc_id, doc in enumerate(sample['documents']):

        # 计算每个doc的 paragraph 的置信度得分
        para_infos = []
        for para_id, para in enumerate(doc['paragraphs']):
            confidence_score = para_confidence_score(para_id, doc['paragraphs'], question)
            # ((段落匹配得分，段落长度)，段落的原始下标)
            para_infos.append((confidence_score, len(para), para_id))

        # 按照 match_score 降序排列，按照段落长度升序排列，以保证较短的置信度高的段落能召回
        para_infos.sort(key=lambda x: (-x[0], x[1]))

        # 依据 max_doc_len 筛选段落
        selected_para_ids = []
        last_para_id = -1
        last_para_cut_char_idx = -1     # 如果超出范围，最后一个段落需要截断的 char id
        selected_char_len = 0       # 拼接的 char 总长度

        for para_info in para_infos:
            para_id, para_len = para_info[2], para_info[1]
            selected_char_len += para_len
            if selected_char_len <= max_doc_len:
                selected_para_ids.append(para_id)
            else:   # 超出 max_doc_len 范围，截取到 max_doc_len 长度，防止筛掉了答案所在的段落
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
