#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/7 18:16
"""
import sys
import json


def combine_short_and_split_long_para(sample, min_para_len=200, max_min_times=1.5):
    """
    对于长度小于 min_para_len 阈值的 para 选择和后面的进行拼接；
    对于长度大于 max_min_times * min_para_len 阈值的 para 进行切分小段落，同时小段落再进行拼接；
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

        # 切分很长的段落
        pid = 0
        splited_paras = []
        while pid < len(concated_paras):
            para = concated_paras[pid]
            if len(para) <= max_min_times * min_para_len:
                splited_paras.append(para)
            else:
                sub_sents = para.split('。')
                sid = 0
                added_sent = ''
                # 拼接很短的段落
                while sid < len(sub_sents):
                    added_sent += sub_sents[sid]
                    if len(added_sent) > min_para_len:
                        splited_paras.append(added_sent)
                        added_sent = ''
                    sid += 1
                if len(added_sent) > 0:
                    # 如果最后一个 para 较短，拼接到前面一个para
                    if len(added_sent) < min_para_len:
                        splited_paras[-1] = splited_paras[-1] + added_sent
                    else:
                        splited_paras.append(added_sent)
            pid += 1

        document['paragraphs'] = concated_paras


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        combine_short_and_split_long_para(sample)
        print(json.dumps(sample, ensure_ascii=False))
