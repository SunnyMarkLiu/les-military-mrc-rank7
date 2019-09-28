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
import fool
from utils.jieba_util import WordSegmentPOSKeywordExtractor
from collections import Counter

jieba_extractor = WordSegmentPOSKeywordExtractor()
print('jieba prepared.')


def text_analysis(sample):
    """
    中文分词，关键词提取，POS标注
    """
    # question
    sample['seg_que'], sample['pos_que'], sample['kw_que'] = jieba_extractor.extract_sentence(sample['question'], 0.6)
    # answer
    if 'answers' in sample:
        sample['seg_ans'] = jieba_extractor.extract_sentence(sample['answers'], None)
    # supporting_paragraph
    if 'supporting_paragraph' in sample:
        sample['seg_support_para'] = jieba_extractor.extract_sentence(sample['supporting_paragraph'], None)

    question_sapns = set(sample['seg_que'])
    for doc in sample['documents']:
        doc['seg_title'], doc['pos_title'], doc['kw_title'] = jieba_extractor.extract_sentence(doc['title'], 0.6)

        # 分词后的结果
        doc['seg'], doc['term_pos'], doc['term_kw'], doc['term_entity'], doc['term_in_que'] = [], [], [], [], []
        # char 的结果
        doc['char_pos'], doc['char_kw'], doc['char_entity'], doc['char_in_que'] = [], [], [], []

        for para in doc['paragraphs']:
            seg, term_pos, term_kw = jieba_extractor.extract_sentence(para, 0.4)
            term_in_que = [int(token in question_sapns) for token in seg]
            doc['seg'].append(seg)
            doc['term_pos'].append(term_pos)
            doc['term_kw'].append(term_kw)
            doc['term_in_que'].append(term_in_que)

            _, ners = fool.analysis(para)
            entities = ners[0]
            # 处理 char 的 entity 边界
            char_i = 0
            entity_i = 0
            char_entity = []
            while char_i < len(para):
                if entity_i == len(entities):
                    char_entity.append('')
                    char_i += 1
                    continue
                if char_i < entities[entity_i][0]:  # 非实体词的 char
                    char_entity.append('')
                    char_i += 1
                elif entities[entity_i][0] <= char_i < entities[entity_i][0] + len(entities[entity_i][3]):
                    char_entity.append(entities[entity_i][2])
                    char_i += 1
                else:
                    entity_i += 1
            doc['char_entity'].append(char_entity)

            # 处理 term 的 entity 边界问题
            # 处理 char 的 pos，keyword，in_que
            char_pos = []
            char_kw = []
            char_in_que = []
            char_pointer = 0
            term_entity = []
            for term_i, term in enumerate(seg):
                char_pos.extend([term_pos[term_i]] * len(term))
                char_kw.extend([term_kw[term_i]] * len(term))
                char_in_que.extend([term_in_que[term_i]] * len(term))

                char_entities = char_entity[char_pointer: char_pointer + len(term)]
                most_freq_entity = Counter(char_entities).most_common()[0][0]
                term_entity.append(most_freq_entity)
                char_pointer += len(term)

            doc['char_pos'].append(char_pos)
            doc['char_kw'].append(char_kw)
            doc['char_in_que'].append(char_in_que)
            doc['term_entity'].append(term_entity)

if __name__ == '__main__':
    gpu = sys.argv[1]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # disable TF debug logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # INFO/warning/ERROR/FATAL
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        text_analysis(sample)
        print(json.dumps(sample, ensure_ascii=False))
