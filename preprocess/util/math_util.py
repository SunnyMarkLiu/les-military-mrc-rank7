#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/6/16 22:37
"""
import numpy as np
from scipy.stats import pearsonr

MISSING_VALUE_NUMERIC = -1


class MathUtil(object):
    """
    Tool of Math
    """

    @staticmethod
    def count_one_bits(x):
        """
        Calculate the number of bits which are 1
        :param x: number which will be calculated
        :return: number of bits in `x`
        """
        n = 0
        while x:
            n += 1 if (x & 0x01) else 0
            x >>= 1
        return n

    @staticmethod
    def int2binarystr(x):
        """
        Convert the number from decimal to binary
        :param x: decimal number
        :return: string represented binary format of `x`
        """
        s = ""
        while x:
            s += "1" if (x & 0x01) else "0"
            x >>= 1
        return s[::-1]

    @staticmethod
    def try_divide(x, y, val=0.0):
        """
        try to divide two numbers
        """
        if y != 0.0:
            val = float(x) / y
        return val

    @staticmethod
    def corr(x, y_train):
        """
        Calculate correlation between specified feature and labels
        :param x: specified feature in numpy
        :param y_train: labels in numpy
        :return: value of correlation
        """
        if MathUtil.dim(x) == 1:
            corr = pearsonr(x.flatten(), y_train)[0]
            if str(corr) == "nan":
                corr = 0.
        else:
            corr = 1.
        return corr

    @staticmethod
    def dim(x):
        d = 1 if len(x.shape) == 1 else x.shape[1]
        return d

    @staticmethod
    def aggregate(data, modes):
        valid_modes = ["size", "mean", "std", "max", "min", "median"]

        if isinstance(modes, str):
            assert modes.lower() in valid_modes, "Wrong aggregation_mode: %s" % modes
            modes = [modes.lower()]
        elif isinstance(modes, list):
            for m in modes:
                assert m.lower() in valid_modes, "Wrong aggregation_mode: %s" % m
                modes = [m.lower() for m in modes]
        aggregators = [getattr(np, m) for m in modes]

        aggeration_value = list()
        for agg in aggregators:
            try:
                s = agg(data)
            except ValueError:
                s = MISSING_VALUE_NUMERIC
            aggeration_value.append(s)
        return aggeration_value

    @staticmethod
    def cut_prob(p):
        p[p > 1.0 - 1e-15] = 1.0 - 1e-15
        p[p < 1e-15] = 1e-15
        return p

    @staticmethod
    def logit(p):
        assert isinstance(p, np.ndarray), 'type error'
        p = MathUtil.cut_prob(p)
        return np.log(p / (1. - p))

    @staticmethod
    def logistic(y):
        return np.exp(y) / (1. + np.exp(y))
