#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/5 15:20
"""
# coding = utf-8
import numpy as np


class RougeL(object):
    def __init__(self, gamma=1.2):
        self.gamma = gamma  # gamma 为常量
        self.inst_scores = []
        self.r_scores = []
        self.p_scores = []

    def _lcs(self, x, y):
        """
        Computes the length of the longest common subsequence (lcs) between two
        strings. The implementation below uses a DP programming algorithm and runs
        in O(nm) time where n = len(x) and m = len(y).
        Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
        Args:
          x: collection of words
          y: collection of words
        Returns:
          Table of dictionary of coord and len lcs
        """
        n, m = len(x), len(y)
        table = dict()
        for i in range(n + 1):
            for j in range(m + 1):
                if i == 0 or j == 0:
                    table[i, j] = 0
                elif x[i - 1] == y[j - 1]:
                    table[i, j] = table[i - 1, j - 1] + 1
                else:
                    table[i, j] = max(table[i - 1, j], table[i, j - 1])

        return table

    def lcs(self, x, y):
        """
        Returns the length of the Longest Common Subsequence between sequences x
        and y.
        Source: http://www.algorithmist.com/index.php/Longest_Common_Subsequence
        Args:
          x: sequence of words
          y: sequence of words
        Returns
          integer: Length of LCS between x and y
        """
        table = self._lcs(x, y)
        n, m = len(x), len(y)
        return table[n, m]

    def add_inst(self, cand: str, ref: str):
        """根据参考答案分析出预测答案的分数
        Arguments:
            cand {str} -- 预测答案
            ref {str} -- 参考答案
        """

        basic_lcs = self.lcs(cand, ref)
        p_denom = len(cand)
        r_denom = len(ref)
        prec = basic_lcs / p_denom if p_denom > 0. else 0.
        rec = basic_lcs / r_denom if r_denom > 0. else 0.
        if prec != 0 and rec != 0:
            score = ((1 + self.gamma ** 2) * prec * rec) / \
                float(rec + self.gamma**2 * prec)
        else:
            score = 0
        self.inst_scores.append(score)
        self.r_scores.append(rec)
        self.p_scores.append(prec)
        return self

    def get_score(self) -> float:
        """计算cand预测数据的RougeL分数
        Returns:
            float -- RougeL分数
        """
        return 1. * sum(self.inst_scores) / len(self.inst_scores)
