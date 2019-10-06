#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/6/16 22:29
"""
import lzma
from difflib import SequenceMatcher
from util.math_util import MathUtil
from util import levenshtein
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
from scipy import spatial
from fuzzywuzzy import fuzz
import numpy as np
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
bleu_smoothing_function = SmoothingFunction().method1


class DistanceUtil(object):
    """
    Tool of Distance
    """

    @staticmethod
    def levenshtein_1(str1, str2):
        """
        levenshtein distance shortest alignment
        The normalized distance will be a float between 0 and 1, where 0 means equal and 1 completely different.
        """
        res = levenshtein.nlevenshtein(str1, str2, method=1)
        return res

    @staticmethod
    def levenshtein_2(str1, str2):
        """
        levenshtein distance longest alignment
        The normalized distance will be a float between 0 and 1, where 0 means equal and 1 completely different.
        """
        res = levenshtein.nlevenshtein(str1, str2, method=2)
        return res

    @staticmethod
    def is_str_match(str1, str2, threshold=1.0):
        assert 0.0 <= threshold <= 1.0, "Wrong threshold."
        if float(threshold) == 1.0:
            return str1 == str2
        else:
            return (1. - DistanceUtil.edit_dist(str1, str2)) >= threshold

    @staticmethod
    def longest_match_size(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return match.size

    @staticmethod
    def longest_match_ratio(str1, str2):
        sq = SequenceMatcher(lambda x: x == " ", str1, str2)
        match = sq.find_longest_match(0, len(str1), 0, len(str2))
        return MathUtil.try_divide(match.size, min(len(str1), len(str2)))

    @staticmethod
    def compression_dist(x, y, l_x=None, l_y=None):
        if x == y:
            return 0
        x_b = x.encode('utf-8')
        y_b = y.encode('utf-8')
        if l_x is None:
            l_x = len(lzma.compress(x_b))
            l_y = len(lzma.compress(y_b))
        l_xy = len(lzma.compress(x_b + y_b))
        l_yx = len(lzma.compress(y_b + x_b))
        dist = MathUtil.try_divide(min(l_xy, l_yx) - min(l_x, l_y), max(l_x, l_y))
        return dist

    @staticmethod
    def cosine_sim(vec1, vec2):
        try:
            s = cosine_similarity(vec1.reshape(1, -1), vec2.reshape(1, -1))[0][0]
        except:
            try:
                s = cosine_similarity(vec1, vec2)[0][0]
            except:
                s = -1
        return s

    @staticmethod
    def jaccard_coef(str1, str2):
        if not isinstance(str1, set):
            str1 = set(str1)
        if not isinstance(str2, set):
            str2 = set(str2)
        return MathUtil.try_divide(float(len(str1.intersection(str2))), len(str1.union(str2)))

    @staticmethod
    def dice_dist(str1, str2):
        if not isinstance(str1, set):
            str1 = set(str1)
        if not isinstance(str2, set):
            str2 = set(str2)
        return MathUtil.try_divide(2. * float(len(str1.intersection(str2))), (len(str1) + len(str2)))

    @staticmethod
    def countbased_cos_distance(tokenization1, tokenization2):
        """
        基于计数的 cos 距离
        """
        def build_vector(iterable1, iterable2):
            counter1 = Counter(iterable1)
            counter2 = Counter(iterable2)
            all_items = set(counter1.keys()).union(set(counter2.keys()))
            vector1 = [counter1[k] for k in all_items]
            vector2 = [counter2[k] for k in all_items]
            vector1 = [1e-6] if len(vector1) == 0 else vector1
            vector2 = [1e-6] if len(vector2) == 0 else vector2
            return vector1, vector2

        v1, v2 = build_vector(tokenization1, tokenization2)
        dist = 1 - spatial.distance.cosine(v1, v2)
        return dist

    @staticmethod
    def fuzzy_matching_ratio(str1, str2, ratio_func='partial_ratio'):
        """
        字符串模糊匹配
        :param str1: 字符串
        :param str2: 字符串
        :param ratio_func: ratio, partial_ratio, token_sort_ratio, token_set_ratio
        """
        if ratio_func == 'ratio':
            # Normalize to [0 - 1] range.
            return fuzz.ratio(str1, str2) / 100.0
        if ratio_func == 'partial_ratio':
            return fuzz.partial_ratio(str1, str2) / 100.0
        if ratio_func == 'token_sort_ratio':
            return fuzz.token_sort_ratio(str1, str2) / 100.0
        if ratio_func == 'token_set_ratio':
            return fuzz.token_set_ratio(str1, str2) / 100.0

    @staticmethod
    def word_match_share(str1, str2):
        """
        The total number of words in both sentences
        """
        str1_words, str2_words = {}, {}
        for word in str1:
            str1_words[word] = 1
        for word in str2:
            str2_words[word] = 1
        shared_words_in_q1 = [w for w in str1_words.keys() if w in str2_words]
        shared_words_in_q2 = [w for w in str2_words.keys() if w in str1_words]
        r = (len(shared_words_in_q1) + len(shared_words_in_q2)) / (len(str1_words) + len(str2_words))
        return r

    @staticmethod
    def calc_word_ngram_distance(str1, str2, ngram):
        """
        基于 n-gram 的各种距离特征
        """

        def get_words_ngrams(seq, _n):
            return [seq[j:j + _n] for j in range(len(seq) - _n + 1)]

        str1_ngram = get_words_ngrams(str1, ngram)
        str2_ngram = get_words_ngrams(str2, ngram)

        cut_len = min((len(str1_ngram), len(str2_ngram)))
        cos_distances = []
        levenshtein_distances = []
        for i in range(cut_len):
            cos_distance = DistanceUtil.countbased_cos_distance(str1_ngram[i], str2_ngram[i])
            levenshtein_distance = DistanceUtil.levenshtein_1(str1_ngram[i], str2_ngram[i])
            cos_distances.append(cos_distance)
            levenshtein_distances.append(levenshtein_distance)

        cos_distances = [0.0] if len(cos_distances) == 0 else cos_distances
        levenshtein_distances = [0.0] if len(levenshtein_distances) == 0 else levenshtein_distances

        return np.mean(cos_distances), np.mean(levenshtein_distances)

    @staticmethod
    def f1_score(str1, str2):
        common = Counter(str1) & Counter(str2)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        p = 1.0 * num_same / len(str1)
        r = 1.0 * num_same / len(str2)
        f1 = (2 * p * r) / (p + r)
        return f1

    @staticmethod
    def bleu_score(str1, str2):
        return sentence_bleu(references=str1, hypothesis=str2, smoothing_function=bleu_smoothing_function)
