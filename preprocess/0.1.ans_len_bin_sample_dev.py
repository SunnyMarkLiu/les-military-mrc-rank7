#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
根据 answer 长度分布，进行 bin 切分，从 bin 中进行采样得到 dev

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/10/6 21:11
"""
import pandas as pd


def same_freq_bincut(series, n):
    edages = pd.Series([i / n for i in range(n)])  # 转换成百分比
    func = lambda x: (edages >= x).values.argmax()  # 函数：(edages >= x)返回fasle/true列表中第一次出现true的索引值
    return series.rank(pct=1).astype(float).apply(func)


train_df = pd.read_csv('../input/original/train_round_0.csv', sep=',')
train_df['answer_str'] = train_df['answer'].str.replace('@content\d@', '')
train_df['answer_len'] = train_df['answer_str'].str.len()

answer_len_cut_bins = 25
dev_ratio = 0.1

# 对训练集进行等频 bin 切分
sample_belong_bins = same_freq_bincut(train_df['answer_len'], answer_len_cut_bins)
# 按照不同的 bin 进行划分训练集
bin_cut_train_sets = [[] for _ in range(answer_len_cut_bins)]
for i, sample in train_df.iterrows():
    answer_bin = sample_belong_bins[i]
    bin_cut_train_sets[answer_bin].append(sample['question_id'])

dev_sample_ques_ids = []
for i in range(answer_len_cut_bins):
    dev_sample_ques_ids.extend(bin_cut_train_sets[i][: int(dev_ratio * len(bin_cut_train_sets[i]))])

print('sampled dev count:', len(dev_sample_ques_ids))
with open('{}_dev_sample_ques_ids.txt'.format(len(dev_sample_ques_ids)), 'w') as f:
    f.write(','.join(dev_sample_ques_ids))
