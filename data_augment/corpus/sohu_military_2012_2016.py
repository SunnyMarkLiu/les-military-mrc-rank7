#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/13 22:32
"""
import pandas as pd

military_corpus_fout = open('../../input/military_corpus/sohu_military_2012_2016.txt', 'w', encoding='utf8')

df = pd.read_csv('../../input/military_corpus/sohu_military_2012_2016.csv')
for content in df['content']:
    sents = [sent + '。\n' for sent in content.split('。') if len(sent) > 3]
    sents.append('\n')
    military_corpus_fout.writelines(sents)
