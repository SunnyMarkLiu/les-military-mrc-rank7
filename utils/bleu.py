#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/5 15:20
"""
import math
from utils import common


class Bleu(object):
    def __init__(self, n_size=4):
        self.match_ngram = {}
        self.candi_ngram = {}
        self.bp_r = 0
        self.bp_c = 0
        self.n_size = n_size

    def add_inst(self, cand: str, ref: str):
        """根据添加的预测答案和参考答案，更新match_gram和candi_gram
        Arguments:
            cand {str} -- 预测答案
            ref {str} -- 参考答案
        """

        for n_size in range(self.n_size):
            self.count_ngram(cand, ref, n_size + 1)
        self.count_bp(cand, ref)
        return self

    def count_ngram(self, cand: str, ref: str, n_size: int):
        """计算子序列重合的个数，并存储到字典中
        Arguments:
            cand {str} -- 预备答案
            ref {str} -- 参考答案
            n_size {int} -- 子序列的大小
        """

        cand_ngram = common.get_ngram(cand, n_size)
        ref_ngram = common.get_ngram(ref, n_size)
        if n_size not in self.match_ngram:
            self.match_ngram[n_size] = 0
            self.candi_ngram[n_size] = 0
        match_size, cand_size = common.get_match_size(cand_ngram, ref_ngram)
        self.match_ngram[n_size] += match_size
        self.candi_ngram[n_size] += cand_size

    def count_bp(self, cand: str, ref: str):
        """计算BP参数对应的r和c
        Arguments:
            cand {str} -- 预备答案
            ref {str} -- 参考答案
        Returns:
            float -- BP参数计算结果
        """

        self.bp_c += len(cand)
        self.bp_r += len(ref)

    def get_score(self) -> float:
        """计算字符串cand的Bleu分数, 并返回
        Returns:
            bleu_score {float} -- bleu分数
        """
        prob_list = [
            self.match_ngram[n_size + 1] / float(self.candi_ngram[n_size + 1]) if self.candi_ngram[
                                                                                      n_size + 1] != 0 else 0.0
            for n_size in range(self.n_size)
        ]
        bleu_score = prob_list[0]
        for n in range(1, self.n_size):
            bleu_score *= prob_list[n]
        bleu_score = bleu_score ** (1. / float(self.n_size))
        bp = math.exp(min(1 - self.bp_r / float(self.bp_c), 0))
        bleu_score = bp * bleu_score
        # print('bleu score: {}'.format(bleu_score))
        return bleu_score
