#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/10/10 20:42
"""
import torch
from torch.utils.data import Dataset


# 特征展开
def flat_feature_list(input_):
    index = []
    for item_ in input_:
        index += [item_[0]] * item_[1]
    return index


class LazyLoadTensorDataset(Dataset):

    def __init__(self, features, is_training):
        """
        Args:
            features: feature list
        """
        self.features = features
        self.is_training = is_training

    def __getitem__(self, index):
        feature = self.features[index]

        input_ids = torch.tensor(feature['input_ids'], dtype=torch.long)
        input_mask = torch.tensor(flat_feature_list(feature['input_mask']), dtype=torch.long)
        segment_ids = torch.tensor(flat_feature_list(feature['segment_ids']), dtype=torch.long)
        p_mask = torch.tensor(flat_feature_list(feature['p_mask']), dtype=torch.long)
        doc_position = torch.tensor(feature['doc_position'], dtype=torch.long)
        char_pos = torch.tensor(flat_feature_list(feature['char_pos']), dtype=torch.long)
        char_kw = torch.tensor(flat_feature_list(feature['char_kw']), dtype=torch.long)
        char_in_que = torch.tensor(flat_feature_list(feature['char_in_que']), dtype=torch.long)
        char_entity = torch.tensor(flat_feature_list(feature['char_entity']), dtype=torch.long)
        fuzzy_matching_ratio = torch.tensor(flat_feature_list(feature['fuzzy_matching_ratio']), dtype=torch.float)
        fuzzy_matching_partial_ratio = torch.tensor(flat_feature_list(feature['fuzzy_matching_partial_ratio']), dtype=torch.float)
        fuzzy_matching_token_sort_ratio = torch.tensor(flat_feature_list(feature['fuzzy_matching_token_sort_ratio']), dtype=torch.float)
        fuzzy_matching_token_set_ratio = torch.tensor(flat_feature_list(feature['fuzzy_matching_token_set_ratio']), dtype=torch.float)
        word_match_share = torch.tensor(flat_feature_list(feature['word_match_share']), dtype=torch.float)
        f1_score = torch.tensor(flat_feature_list(feature['f1_score']), dtype=torch.float)
        mean_cos_dist_2gram = torch.tensor(flat_feature_list(feature['mean_cos_dist_2gram']), dtype=torch.float)
        mean_leve_dist_2gram = torch.tensor(flat_feature_list(feature['mean_leve_dist_2gram']), dtype=torch.float)
        mean_cos_dist_3gram = torch.tensor(flat_feature_list(feature['mean_cos_dist_3gram']), dtype=torch.float)
        mean_leve_dist_3gram = torch.tensor(flat_feature_list(feature['mean_leve_dist_3gram']), dtype=torch.float)
        mean_cos_dist_4gram = torch.tensor(flat_feature_list(feature['mean_cos_dist_4gram']), dtype=torch.float)
        mean_leve_dist_4gram = torch.tensor(flat_feature_list(feature['mean_leve_dist_4gram']), dtype=torch.float)
        mean_cos_dist_5gram = torch.tensor(flat_feature_list(feature['mean_cos_dist_5gram']), dtype=torch.float)
        mean_leve_dist_5gram = torch.tensor(flat_feature_list(feature['mean_leve_dist_5gram']), dtype=torch.float)

        tensors = [input_ids, input_mask, segment_ids,
                p_mask,
                doc_position,
                char_pos,
                char_kw,
                char_in_que,
                fuzzy_matching_ratio,
                fuzzy_matching_partial_ratio,
                fuzzy_matching_token_sort_ratio,
                fuzzy_matching_token_set_ratio,
                word_match_share,
                f1_score,
                mean_cos_dist_2gram,
                mean_leve_dist_2gram,
                mean_cos_dist_3gram,
                mean_leve_dist_3gram,
                mean_cos_dist_4gram,
                mean_leve_dist_4gram,
                mean_cos_dist_5gram,
                mean_leve_dist_5gram,
                char_entity]

        if self.is_training:
            start_positions = torch.tensor(feature['start_position'], dtype=torch.long)
            end_positions = torch.tensor(feature['end_position'], dtype=torch.long)

            tensors.append(start_positions)
            tensors.append(end_positions)
        else:
            example_index = torch.tensor(index, dtype=torch.long)
            tensors.append(example_index)

        return tensors

    def __len__(self):
        return len(self.features)
