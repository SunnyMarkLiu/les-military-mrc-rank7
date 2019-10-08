#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
构建匹配、距离特征

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/24 21:52
"""
import sys
import json
import re
from util.distance_util import DistanceUtil


def extract_match_features(sample):
    que_str = sample['question']

    for doc in sample['documents']:
        sents = re.split('[，。！]', doc['content'])

        doc['levenshtein_dist'], doc['longest_match_size'], doc['longest_match_ratio'] = [], [], []
        doc['compression_dist'], doc['jaccard_coef'], doc['dice_dist'], doc['countbased_cos_distance'] = [], [], [], []
        doc['fuzzy_matching_ratio'], doc['fuzzy_matching_partial_ratio'], doc['fuzzy_matching_token_sort_ratio'] = [], [], []
        doc['fuzzy_matching_token_set_ratio'], doc['word_match_share'], doc['f1_score'] = [], [], []
        doc['mean_cos_dist_2gram'], doc['mean_leve_dist_2gram'], doc['mean_cos_dist_3gram'], doc['mean_leve_dist_3gram'] = [], [], [], []
        doc['mean_cos_dist_4gram'], doc['mean_leve_dist_4gram'], doc['mean_cos_dist_5gram'], doc['mean_leve_dist_5gram'] = [], [], [], []

        doc['sent_lens'] = []
        for i, sent in enumerate(sents):
            # 计算分割的句子和问题的距离特征
            sent_len = len(sent) + 1
            if i == len(sents) - 1:
                if doc['content'][-1] in {'，', '。', '！'}:
                    sent_len -= 1
            doc['sent_lens'].append(sent_len)

            doc['levenshtein_dist'].append(DistanceUtil.levenshtein_1(sent, que_str))
            doc['longest_match_size'].append(DistanceUtil.longest_match_size(sent, que_str))
            doc['longest_match_ratio'].append(DistanceUtil.longest_match_ratio(sent, que_str))
            doc['compression_dist'].append(DistanceUtil.compression_dist(sent, que_str))
            doc['jaccard_coef'].append(DistanceUtil.jaccard_coef(sent, que_str))
            doc['dice_dist'].append(DistanceUtil.dice_dist(sent, que_str))
            doc['countbased_cos_distance'].append(DistanceUtil.countbased_cos_distance(sent, que_str))
            doc['fuzzy_matching_ratio'].append(DistanceUtil.fuzzy_matching_ratio(sent, que_str, ratio_func='ratio'))
            doc['fuzzy_matching_partial_ratio'].append(DistanceUtil.fuzzy_matching_ratio(sent, que_str, ratio_func='partial_ratio'))
            doc['fuzzy_matching_token_sort_ratio'].append(DistanceUtil.fuzzy_matching_ratio(sent, que_str, ratio_func='token_sort_ratio'))
            doc['fuzzy_matching_token_set_ratio'].append(DistanceUtil.fuzzy_matching_ratio(sent, que_str, ratio_func='token_set_ratio'))
            doc['word_match_share'].append(DistanceUtil.word_match_share(sent, que_str))
            doc['f1_score'].append(DistanceUtil.f1_score(sent, que_str))

            mean_cos_dist_2gram, mean_leve_dist_2gram = DistanceUtil.calc_word_ngram_distance(sent, que_str, ngram=2)
            doc['mean_cos_dist_2gram'].append(mean_cos_dist_2gram)
            doc['mean_leve_dist_2gram'].append(mean_leve_dist_2gram)
            mean_cos_dist_3gram, mean_leve_dist_3gram = DistanceUtil.calc_word_ngram_distance(sent, que_str, ngram=3)
            doc['mean_cos_dist_3gram'].append(mean_cos_dist_3gram)
            doc['mean_leve_dist_3gram'].append(mean_leve_dist_3gram)
            mean_cos_dist_4gram, mean_leve_dist_4gram = DistanceUtil.calc_word_ngram_distance(sent, que_str, ngram=4)
            doc['mean_cos_dist_4gram'].append(mean_cos_dist_4gram)
            doc['mean_leve_dist_4gram'].append(mean_leve_dist_4gram)
            mean_cos_dist_5gram, mean_leve_dist_5gram = DistanceUtil.calc_word_ngram_distance(sent, que_str, ngram=5)
            doc['mean_cos_dist_5gram'].append(mean_cos_dist_5gram)
            doc['mean_leve_dist_5gram'].append(mean_leve_dist_5gram)

if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        extract_match_features(sample)
        print(json.dumps(sample, ensure_ascii=False))
