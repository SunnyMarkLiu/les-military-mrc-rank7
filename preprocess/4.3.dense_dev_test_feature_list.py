#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/10/9 08:46
"""
import json
import sys

def dense_feature_list(feat_list):
    feat_list = [0 if x == 'NaN' else x for x in feat_list]
    feat_len_list = [(None, 0)]
    i = 0
    while i < len(feat_list):
        if feat_list[i] != feat_len_list[-1][0]:  # 新的特征值
            feat_len_list.append((feat_list[i], 1))
        else:
            feat_len_list[-1] = (feat_len_list[-1][0], feat_len_list[-1][1] + 1)
        i += 1
    return feat_len_list[1:]

def split_features(doc, window_start_idx, window_end_idx):
    """
    对特征列表进行裁剪

    Args:
        doc: dict
        window_start_idx: 包含
        window_end_idx: 不包含
    """
    if 'supported_para_mask' in doc:
        del doc['supported_para_mask']

    doc['content'] = doc['content'][window_start_idx: window_end_idx]

    doc_len = min(window_end_idx - window_start_idx, len(doc['content']))
    doc['char_pos'] = doc['char_pos'][window_start_idx:window_end_idx]
    doc['char_pos'] = dense_feature_list(doc['char_pos'])
    assert sum([l[1] for l in doc['char_pos']]) == doc_len

    doc['char_kw'] = doc['char_kw'][window_start_idx:window_end_idx]
    doc['char_kw'] = dense_feature_list(doc['char_kw'])
    assert sum([l[1] for l in doc['char_kw']]) == doc_len

    doc['char_in_que'] = doc['char_in_que'][window_start_idx:window_end_idx]
    doc['char_in_que'] = dense_feature_list(doc['char_in_que'])
    assert sum([l[1] for l in doc['char_in_que']]) == doc_len

    doc['char_entity'] = doc['char_entity'].split(',')[window_start_idx:window_end_idx]
    doc['char_entity'] = dense_feature_list(doc['char_entity'])
    assert sum([l[1] for l in doc['char_entity']]) == doc_len

    # 特征截断
    for f in ['fuzzy_matching_ratio', 'fuzzy_matching_partial_ratio',
              'fuzzy_matching_token_sort_ratio', 'fuzzy_matching_token_set_ratio', 'word_match_share', 'f1_score',
              'mean_cos_dist_2gram', 'mean_leve_dist_2gram', 'mean_cos_dist_3gram', 'mean_leve_dist_3gram',
              'mean_cos_dist_4gram', 'mean_leve_dist_4gram', 'mean_cos_dist_5gram', 'mean_leve_dist_5gram']:
        feat_list = []
        len_bug = sum(doc['sent_lens']) != doc_len
        for sent_i, sent_len in enumerate(doc['sent_lens']):
            if sent_i == len(doc['sent_lens']) - 1 and len_bug:
                sent_len -= 1
            feat_list.extend([doc[f][sent_i]] * sent_len)

        cut_feat_list = feat_list[window_start_idx: window_end_idx]
        doc[f] = dense_feature_list(cut_feat_list)
        assert sum([l[1] for l in doc[f]]) == doc_len


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())

        sample['ques_char_pos'] = dense_feature_list(sample['ques_char_pos'])
        sample['ques_char_kw'] = dense_feature_list(sample['ques_char_kw'])
        sample['ques_char_in_que'] = dense_feature_list(sample['ques_char_in_que'])
        sample['ques_char_entity'] = dense_feature_list(sample['ques_char_entity'].split(','))

        for doc in sample['documents']:
            split_features(doc, 0, len(doc['content']))

        print(json.dumps(sample, ensure_ascii=False))
