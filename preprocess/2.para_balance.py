#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/7 18:16
"""
import sys
sys.path.append('../')
import json

def combine_short_and_split_long_para(sample, min_para_len=100, max_para_len=2000):
    """
    注意到：需要拼接的较短的段落一般末尾没有句号。

    答案绝对匹配下，找到答案的样本占比：
    min_para_len=100， max_para_len=+inf，ceil rougel=1: 0.6976210602584229

    min_para_len=100， max_para_len= 400，ceil rougel=1: 0.6698063840920984
    min_para_len=100， max_para_len= 600，ceil rougel=1: 0.6805136255685706
    min_para_len=100， max_para_len= 800，ceil rougel=1: 0.6856659823692791
    min_para_len=100， max_para_len=1000，ceil rougel=1: 0.6898120194823492
    min_para_len=100， max_para_len=1200，ceil rougel=1: 0.6920661755826591
    min_para_len=100， max_para_len=1400，ceil rougel=1: 0.6934750231453528
    min_para_len=100， max_para_len=1600，ceil rougel=1: 0.6945215956204968
    min_para_len=100， max_para_len=1800，ceil rougel=1: 0.6951656402205852
    min_para_len=100， max_para_len=2000，ceil rougel=1: 0.6956084208831461 √

    After paragraph extraction (随机选择100个样本测试):
    min_para_len= 30， max_para_len=2000，ceil rougel=1: 0.69
    min_para_len= 50， max_para_len=2000，ceil rougel=1: 0.69
    min_para_len= 80， max_para_len=2000，ceil rougel=1: 0.70
    min_para_len= 90， max_para_len=2000，ceil rougel=1: 0.69
    min_para_len=100， max_para_len=2000，ceil rougel=1: 0.70 √
    min_para_len=120， max_para_len=2000，ceil rougel=1: 0.70
    min_para_len=150， max_para_len=2000，ceil rougel=1: 0.71
    min_para_len=200， max_para_len=2000，ceil rougel=1: 0.68

    Args:
        min_para_len: 合法段落的最小长度，小于该长度的需要合并
        max_para_len: 合法段落的最大长度，超出该范围的需要拆分
    """
    for document in sample['documents']:
        concated_paras = []
        pid = 0
        added_para = ''
        # 拼接很短的段落
        while pid < len(document['paragraphs']):
            added_para += document['paragraphs'][pid]
            if len(added_para) > min_para_len:
                concated_paras.append(added_para)
                added_para = ''
            pid += 1
        if len(added_para) > 0:  # 注意加上最后一个para
            concated_paras.append(added_para)

        document['paragraphs'] = concated_paras

        # 统计段落的长度分布，以及答案是否在长段落的情况，决定是否需要对长段落进行拆分
        # 原始 raw 样本中，答案在长度超过800的段落的样本有 5946 个，其中 1014 个样本的答案在长度超过800的段落中，
        # 因此有必要按照某种策略进行长段落的拆分！

        # 针对问题，找到在长段落和问题最大匹配的下标，决定切分的位置 (计算复杂度过大)
        splited_paras = []
        for para in document['paragraphs']:
            if len(para) > max_para_len:
                sents = para.split('。')

                sent_i = 0
                added_para = ''
                while sent_i < len(sents):
                    if len(added_para) > max_para_len:
                        splited_paras.append(added_para)
                        added_para = ''
                    else:
                        added_para += sents[sent_i] + '。'
                    sent_i += 1
                if len(added_para) > 0:  # 注意加上最后一个para
                    splited_paras.append(added_para)
            else:
                splited_paras.append(para)

        document['paragraphs'] = splited_paras

if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        combine_short_and_split_long_para(sample)
        print(json.dumps(sample, ensure_ascii=False))
