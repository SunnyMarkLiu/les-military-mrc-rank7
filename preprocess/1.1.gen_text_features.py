#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
分词和关键词、POS 标注

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/6 09:19
"""
import os
import sys

sys.path.append('../')
import sys
import json
from utils.jieba_util import WordSegmentPOSKeywordExtractor

jieba_extractor = WordSegmentPOSKeywordExtractor()
print('jieba prepared.')

def extract_text_features(text, question_sapns):
    seg, term_pos, term_kw = jieba_extractor.extract_sentence(text, 0.4)
    term_in_que = [int(token in question_sapns) for token in seg]
    # 处理 term 的 entity 边界问题
    # 处理 char 的 pos，keyword，in_que
    char_pos, char_kw, char_in_que = [], [], []
    char_pointer = 0
    for term_i, term in enumerate(seg):
        char_pos.extend([term_pos[term_i]] * len(term))
        char_kw.extend([term_kw[term_i]] * len(term))
        char_in_que.extend([term_in_que[term_i]] * len(term))
        char_pointer += len(term)

    return char_pos, char_kw, char_in_que


def text_analysis(sample):
    """
    中文分词，关键词提取，POS标注
    """
    # question
    char_pos, char_kw, char_in_que = extract_text_features(text=sample['question'], question_sapns=set())
    sample['ques_char_pos'] = char_pos
    sample['ques_char_kw'] = char_kw
    sample['ques_char_in_que'] = char_in_que

    question_sapns = set(sample['question'])
    for doc in sample['documents']:
        char_pos, char_kw, char_in_que = extract_text_features(text=doc['content'], question_sapns=question_sapns)
        doc['char_pos'] = char_pos
        doc['char_kw'] = char_kw
        doc['char_in_que'] = char_in_que

if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        text_analysis(sample)
        print(json.dumps(sample, ensure_ascii=False))
