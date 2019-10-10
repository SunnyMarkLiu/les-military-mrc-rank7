
# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
""" Load SQuAD dataset. """

from __future__ import absolute_import, division, print_function

import json
import logging
import math
import collections
from io import open
import compress_pickle

from pytorch_transformers.tokenization_bert import BasicTokenizer

# Required by XLNet evaluation method to compute optimal threshold (see write_predictions_extended() method)
# from utils_les_evaluate import find_all_best_thresh_v2, make_qid_to_has_ans, get_raw_scores
import random
logger = logging.getLogger(__name__)


# 任务定义
ANSWER_MRC = "answer_mrc"
BRIDGE_ENTITY_MRC = "bridge_entity_mrc"

# POS和NER映射
POS_DIM = 30
POS2ID = {'blank': 0, 'nrt': 1, 'eng': 2, 'n': 3, 'f': 4, 'yg': 5, 'nt': 6, 'rr': 7, 'ad': 8, 'nr': 9, 'dg': 10,
          't': 11, 'bg': 12, 'ag': 13, 'ns': 14, 'an': 15, 'b': 16, 'm': 17,
          'v': 18, 'x': 19, 'q': 20, 'tg': 21, 'nz': 22, 'mq': 23, 'nrfg': 24, 'a': 25, 'i': 26,
          'mg': 27, 's': 28, 'other': 29}
NER_DIM = 7
NER2ID = {'': 0, 'C': 1, 'J': 2, 'L': 3, 'O': 4, 'P': 5, 'T': 6}


class SquadExample(object):
    """
    A single training/test example for the Squad dataset.
    For examples without an answer, the start and end position are -1.
    """

    def __init__(self,
                 qas_id,
                 question_text,
                 doc_tokens,
                 orig_answer_text=None,
                 start_position=None,
                 end_position=None,
                 is_impossible=None,
                 doc_position=None,
                 ques_char_pos=None,
                 ques_char_kw=None,
                 ques_char_entity=None,
                 char_pos=None,
                 char_kw=None,
                 char_in_que=None,
                 fuzzy_matching_ratio=None,
                 fuzzy_matching_partial_ratio=None,
                 fuzzy_matching_token_sort_ratio=None,
                 fuzzy_matching_token_set_ratio=None,
                 word_match_share=None,
                 f1_score=None,
                 mean_cos_dist_2gram=None,
                 mean_leve_dist_2gram=None,
                 mean_cos_dist_3gram=None,
                 mean_leve_dist_3gram=None,
                 mean_cos_dist_4gram=None,
                 mean_leve_dist_4gram=None,
                 mean_cos_dist_5gram=None,
                 mean_leve_dist_5gram=None,
                 char_entity=None):
        self.qas_id = qas_id
        self.question_text = question_text
        self.doc_tokens = doc_tokens
        self.orig_answer_text = orig_answer_text
        self.start_position = start_position
        self.end_position = end_position
        self.is_impossible = is_impossible

        # 额外特征
        self.doc_position = doc_position
        self.ques_char_pos = ques_char_pos
        self.ques_char_kw = ques_char_kw
        self.ques_char_entity = ques_char_entity
        self.char_pos = char_pos
        self.char_kw = char_kw
        self.char_in_que = char_in_que
        self.fuzzy_matching_ratio = fuzzy_matching_ratio
        self.fuzzy_matching_partial_ratio = fuzzy_matching_partial_ratio
        self.fuzzy_matching_token_sort_ratio = fuzzy_matching_token_sort_ratio
        self.fuzzy_matching_token_set_ratio = fuzzy_matching_token_set_ratio
        self.word_match_share = word_match_share
        self.f1_score = f1_score
        self.mean_cos_dist_2gram = mean_cos_dist_2gram
        self.mean_leve_dist_2gram = mean_leve_dist_2gram
        self.mean_cos_dist_3gram = mean_cos_dist_3gram
        self.mean_leve_dist_3gram = mean_leve_dist_3gram
        self.mean_cos_dist_4gram = mean_cos_dist_4gram
        self.mean_leve_dist_4gram = mean_leve_dist_4gram
        self.mean_cos_dist_5gram = mean_cos_dist_5gram
        self.mean_leve_dist_5gram = mean_leve_dist_5gram
        self.char_entity = char_entity

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        s = ""
        s += "qas_id: %s" % (self.qas_id)
        s += ", question_text: %s" % (
            self.question_text)
        s += ", doc_tokens: [%s]" % (" ".join(self.doc_tokens))
        if self.start_position:
            s += ", start_position: %d" % (self.start_position)
        if self.end_position:
            s += ", end_position: %d" % (self.end_position)
        if self.is_impossible:
            s += ", is_impossible: %r" % (self.is_impossible)
        return s


def read_squad_examples(task_name, input_file, is_training, version_2_with_negative):
    """Read a SQuAD json file into a list of SquadExample."""

    def is_whitespace(c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False

    examples = []
    with open(input_file) as fin:
        log_steps = 5000  # log打印间隔
        for line_id, line in enumerate(fin):
            line = line.strip()

            if log_steps > 0 and (line_id + 1) % log_steps == 0:
                logger.info('We have read {} lines through les-json-data'.format(line_id + 1))
            if not line:
                logger.warning('There is an empty line in les-json-data, line_id={}'.format(line_id + 1))
                continue

            sample = json.loads(line)
            question_text = sample['question']
            context_num = len(sample['documents'])  # 莱斯杯固定都是5个
            context_list = [doc['content'] for doc in sample['documents']]

            # 处理question特征
            ques_char_pos = sample['ques_char_pos']
            ques_char_kw = sample['ques_char_kw']
            ques_char_entity = sample['ques_char_entity']
            for item_ in ques_char_pos:
                item_[0] = POS2ID[item_[0]] if item_[0] in POS2ID else POS2ID['other']
            for item_ in ques_char_entity:
                item_[0] = NER2ID[item_[0]]

            if is_training:
                # 注：answer_labels字段和bridging_entity_labels字段均代表(docid, start, end)

                # answer task不存在没答案的sample，bridge entity task则存在
                if task_name == ANSWER_MRC:
                    if len(sample['answer_labels']) == 0:
                        logger.warning('There is an empty answer training sample in les-json-data, line_id={}'.format(line_id + 1))
                        continue
                    match_doc_ids = [label[0] for label in sample['answer_labels']]
                    sample['labels'] = [[label[1], label[2]] for label in sample['answer_labels']]
                else:
                    no_bridging_entity_label = len(sample['bridging_entity_labels']) == 0
                    if no_bridging_entity_label:
                        match_doc_ids = []
                        sample['labels'] = []
                    else:
                        # 对于bridge entity其实label只有一个，为了代码统一用list套了一层
                        match_doc_ids = [sample['bridging_entity_labels'][0]]
                        sample['labels'] = [[sample['bridging_entity_labels'][1], sample['bridging_entity_labels'][2]]]

            for doc_id in range(context_num):  # doc_id代表进行到一个例子中的第几个documents了
                doc_tokens = []
                char_to_word_offset = []
                # word与char一一对应
                doc_tokens = list(context_list[doc_id])
                char_to_word_offset = list(range(len(doc_tokens)))

                # document特征
                doc = sample['documents'][doc_id]

                char_pos = doc['char_pos']
                char_kw = doc['char_kw']
                char_in_que = doc['char_in_que']
                fuzzy_matching_ratio = doc['fuzzy_matching_ratio']
                fuzzy_matching_partial_ratio = doc['fuzzy_matching_partial_ratio']
                fuzzy_matching_token_sort_ratio = doc['fuzzy_matching_token_sort_ratio']
                fuzzy_matching_token_set_ratio = doc['fuzzy_matching_token_set_ratio']
                word_match_share = doc['word_match_share']
                f1_score = doc['f1_score']
                mean_cos_dist_2gram = doc['mean_cos_dist_2gram']
                mean_leve_dist_2gram = doc['mean_leve_dist_2gram']
                mean_cos_dist_3gram = doc['mean_cos_dist_3gram']
                mean_leve_dist_3gram = doc['mean_leve_dist_3gram']
                mean_cos_dist_4gram = doc['mean_cos_dist_4gram']
                mean_leve_dist_4gram = doc['mean_leve_dist_4gram']
                mean_cos_dist_5gram = doc['mean_cos_dist_5gram']
                mean_leve_dist_5gram = doc['mean_leve_dist_5gram']
                char_entity = doc['char_entity']

                # 将pos和ner做映射
                for item_ in char_pos:
                    item_[0] = POS2ID[item_[0]] if item_[0] in POS2ID else POS2ID['other']
                for item_ in char_entity:
                    item_[0] = NER2ID[item_[0]]


                if is_training:
                    if doc_id in match_doc_ids:  # 该document有答案
                        count = 0  # 代表同一个document有多少个答案
                        for match_id, (start, end) in zip(match_doc_ids, sample['labels']):
                            if doc_id != match_id: continue
                            # 有些例子的start, end不在有效范围内
                            if start > end or start not in char_to_word_offset or end not in char_to_word_offset:
                                logger.warning('{}##{}##{} has index bug'.format(sample['question_id'], doc_id, count))
                                continue
                            count += 1
                            qas_id = '{}##{}##{}'.format(sample['question_id'], doc_id, count)
                            start_position = start
                            end_position = end
                            orig_answer_text = doc_tokens[start:end + 1]
                            is_impossible = False

                            example = SquadExample(
                                qas_id=qas_id,
                                question_text=question_text,
                                doc_tokens=doc_tokens,
                                orig_answer_text=orig_answer_text,
                                start_position=start_position,
                                end_position=end_position,
                                is_impossible=is_impossible,
                                doc_position=doc_id,
                                ques_char_pos=ques_char_pos,
                                ques_char_kw=ques_char_kw,
                                ques_char_entity=ques_char_entity,
                                char_pos=char_pos,
                                char_kw=char_kw,
                                char_in_que=char_in_que,
                                fuzzy_matching_ratio=fuzzy_matching_ratio,
                                fuzzy_matching_partial_ratio=fuzzy_matching_partial_ratio,
                                fuzzy_matching_token_sort_ratio=fuzzy_matching_token_sort_ratio,
                                fuzzy_matching_token_set_ratio=fuzzy_matching_token_set_ratio,
                                word_match_share=word_match_share,
                                f1_score=f1_score,
                                mean_cos_dist_2gram=mean_cos_dist_2gram,
                                mean_leve_dist_2gram=mean_leve_dist_2gram,
                                mean_cos_dist_3gram=mean_cos_dist_3gram,
                                mean_leve_dist_3gram=mean_leve_dist_3gram,
                                mean_cos_dist_4gram=mean_cos_dist_4gram,
                                mean_leve_dist_4gram=mean_leve_dist_4gram,
                                mean_cos_dist_5gram=mean_cos_dist_5gram,
                                mean_leve_dist_5gram=mean_leve_dist_5gram,
                                char_entity=char_entity)
                            examples.append(example)
                    else:
                        # 训练集中没有答案的document
                        qas_id = '{}##{}##0'.format(sample['question_id'], doc_id)
                        start_position = -1
                        end_position = -1
                        orig_answer_text = ""
                        is_impossible = True

                        example = SquadExample(
                            qas_id=qas_id,
                            question_text=question_text,
                            doc_tokens=doc_tokens,
                            orig_answer_text=orig_answer_text,
                            start_position=start_position,
                            end_position=end_position,
                            is_impossible=is_impossible,
                            doc_position=doc_id,
                            ques_char_pos=ques_char_pos,
                            ques_char_kw=ques_char_kw,
                            ques_char_entity=ques_char_entity,
                            char_pos=char_pos,
                            char_kw=char_kw,
                            char_in_que=char_in_que,
                            fuzzy_matching_ratio=fuzzy_matching_ratio,
                            fuzzy_matching_partial_ratio=fuzzy_matching_partial_ratio,
                            fuzzy_matching_token_sort_ratio=fuzzy_matching_token_sort_ratio,
                            fuzzy_matching_token_set_ratio=fuzzy_matching_token_set_ratio,
                            word_match_share=word_match_share,
                            f1_score=f1_score,
                            mean_cos_dist_2gram=mean_cos_dist_2gram,
                            mean_leve_dist_2gram=mean_leve_dist_2gram,
                            mean_cos_dist_3gram=mean_cos_dist_3gram,
                            mean_leve_dist_3gram=mean_leve_dist_3gram,
                            mean_cos_dist_4gram=mean_cos_dist_4gram,
                            mean_leve_dist_4gram=mean_leve_dist_4gram,
                            mean_cos_dist_5gram=mean_cos_dist_5gram,
                            mean_leve_dist_5gram=mean_leve_dist_5gram,
                            char_entity=char_entity)
                        examples.append(example)
                else:
                    # not training
                    qas_id = '{}##{}##0'.format(sample['question_id'], doc_id)
                    start_position = None
                    end_position = None
                    orig_answer_text = ""
                    is_impossible = False

                    example = SquadExample(
                        qas_id=qas_id,
                        question_text=question_text,
                        doc_tokens=doc_tokens,
                        orig_answer_text=orig_answer_text,
                        start_position=start_position,
                        end_position=end_position,
                        is_impossible=is_impossible,
                        doc_position=doc_id,
                        ques_char_pos=ques_char_pos,
                        ques_char_kw=ques_char_kw,
                        ques_char_entity=ques_char_entity,
                        char_pos=char_pos,
                        char_kw=char_kw,
                        char_in_que=char_in_que,
                        fuzzy_matching_ratio=fuzzy_matching_ratio,
                        fuzzy_matching_partial_ratio=fuzzy_matching_partial_ratio,
                        fuzzy_matching_token_sort_ratio=fuzzy_matching_token_sort_ratio,
                        fuzzy_matching_token_set_ratio=fuzzy_matching_token_set_ratio,
                        word_match_share=word_match_share,
                        f1_score=f1_score,
                        mean_cos_dist_2gram=mean_cos_dist_2gram,
                        mean_leve_dist_2gram=mean_leve_dist_2gram,
                        mean_cos_dist_3gram=mean_cos_dist_3gram,
                        mean_leve_dist_3gram=mean_leve_dist_3gram,
                        mean_cos_dist_4gram=mean_cos_dist_4gram,
                        mean_leve_dist_4gram=mean_leve_dist_4gram,
                        mean_cos_dist_5gram=mean_cos_dist_5gram,
                        mean_leve_dist_5gram=mean_leve_dist_5gram,
                        char_entity=char_entity)
                    examples.append(example)

                # TODO：此处调用 convert_examples_to_features 的代码，实现合并，减少读取所有 examples 所占用的内存！！
                # 同时转换的 features 多于一定数目如10000进行 compress_pickle 压缩处理，load cache 的时候训练读取pickle
    return examples


def convert_examples_to_features(args, examples, tokenizer, max_seq_length,
                                 doc_stride, max_query_length, is_training,
                                 cls_token_at_end=False,
                                 cls_token='[CLS]', sep_token='[SEP]', pad_token=0,
                                 sequence_a_segment_id=0, sequence_b_segment_id=1,
                                 cls_token_segment_id=0, pad_token_segment_id=0,
                                 mask_padding_with_zero=True):
    """Loads a data file into a list of `InputBatch`s."""

    unique_id = 1000000000
    # cnt_pos, cnt_neg = 0, 0
    # max_N, max_M = 1024, 1024
    # f = np.zeros((max_N, max_M), dtype=np.float32)

    # 对一些重要的字符做一个映射
    convert_token_list = {
        '“': '"',
        '”': '"',
        '…': '...',
        '﹤': '<',
        '﹥': '>',
        '‘': "'",
        '’': "'",
        '﹪': '%',
        '―': '-',
        '—': '-',
        '–': '-',
        '﹟': '#',
        '㈠': '一',
        ' ': '[unused1]',  # 保留空格保持长度一致
        '[SKIPPED]': '[UNK]'  # 保持长度一致
    }

    features = []
    unk_tokens_dict = collections.defaultdict(int)  # 记录vocab中找不到的token
    skipped_tokens_dict = collections.defaultdict(int)  # 记录tokenize后被删掉的token
    log_steps = 5000  # log打印间隔
    logger.info('total examples: {}'.format(len(examples)))
    for (example_index, example) in enumerate(examples):
        if log_steps > 0 and (example_index + 1) % log_steps == 0:
            logger.info('we have converted {} examples to features'.format(example_index + 1))

        question_text = example.question_text
        query_tokens = []
        for token in question_text:
            if token in convert_token_list:
                sub_tokens = [convert_token_list[token]]
            else:
                sub_tokens = tokenizer.tokenize(token)
                if '[UNK]' in sub_tokens:
                    unk_tokens_dict[token] += 1
                if len(sub_tokens) == 0:
                    skipped_tokens_dict[token] += 1
                    sub_tokens = [convert_token_list['[SKIPPED]']]  # 为了保持长度不变
            for sub_token in sub_tokens:
                query_tokens.append(sub_token)

        assert len(question_text) == len(query_tokens)

        if len(query_tokens) > max_query_length:
            query_tokens = query_tokens[0:max_query_length]

        tok_to_orig_index = []
        orig_to_tok_index = []
        all_doc_tokens = []
        for (i, token) in enumerate(example.doc_tokens):
            orig_to_tok_index.append(len(all_doc_tokens))
            if token in convert_token_list:
                sub_tokens = [convert_token_list[token]]
            else:
                sub_tokens = tokenizer.tokenize(token)
                if '[UNK]' in sub_tokens:
                    unk_tokens_dict[token] += 1
                if len(sub_tokens) == 0:
                    skipped_tokens_dict[token] += 1
                    sub_tokens = [convert_token_list['[SKIPPED]']]  # 为了保持长度不变
            for sub_token in sub_tokens:
                tok_to_orig_index.append(i)
                all_doc_tokens.append(sub_token)

        # 在这里doc_tokens和all_doc_tokens长度应该完全一样
        assert len(example.doc_tokens) == len(all_doc_tokens)
        for key, value in enumerate(tok_to_orig_index):
            assert key == value
        for key, value in enumerate(orig_to_tok_index):
            assert key == value

        tok_start_position = None
        tok_end_position = None
        if is_training and example.is_impossible:
            tok_start_position = -1
            tok_end_position = -1
        if is_training and not example.is_impossible:
            tok_start_position = orig_to_tok_index[example.start_position]
            if example.end_position < len(example.doc_tokens) - 1:
                tok_end_position = orig_to_tok_index[example.end_position + 1] - 1
            else:
                tok_end_position = len(all_doc_tokens) - 1
            # 中文不需要这一步
            # (tok_start_position, tok_end_position) = _improve_answer_span(
            #     all_doc_tokens, tok_start_position, tok_end_position, tokenizer,
            #     example.orig_answer_text)

        # The -3 accounts for [CLS], [SEP] and [SEP]
        max_tokens_for_doc = max_seq_length - len(query_tokens) - 3

        # We can have documents that are longer than the maximum sequence length.
        # To deal with this we do a sliding window approach, where we take chunks
        # of the up to our max length with a stride of `doc_stride`.
        _DocSpan = collections.namedtuple(  # pylint: disable=invalid-name
            "DocSpan", ["start", "length"])
        doc_spans = []
        start_offset = 0
        while start_offset < len(all_doc_tokens):
            length = len(all_doc_tokens) - start_offset
            if length > max_tokens_for_doc:
                length = max_tokens_for_doc
            doc_spans.append(_DocSpan(start=start_offset, length=length))
            if start_offset + length == len(all_doc_tokens):
                break
            start_offset += min(length, doc_stride)

        # 特征展开
        def get_flat_feat(input_):
            index = []
            for item_ in input_:
                index += [item_[0]] * item_[1]
            return index
        char_pos_flat = get_flat_feat(example.char_pos)
        char_kw_flat = get_flat_feat(example.char_kw)
        char_in_que_flat = get_flat_feat(example.char_in_que)
        fuzzy_matching_ratio_flat = get_flat_feat(example.fuzzy_matching_ratio)
        fuzzy_matching_partial_ratio_flat = get_flat_feat(example.fuzzy_matching_partial_ratio)
        fuzzy_matching_token_sort_ratio_flat = get_flat_feat(example.fuzzy_matching_token_sort_ratio)
        fuzzy_matching_token_set_ratio_flat = get_flat_feat(example.fuzzy_matching_token_set_ratio)
        word_match_share_flat = get_flat_feat(example.word_match_share)
        f1_score_flat = get_flat_feat(example.f1_score)
        mean_cos_dist_2gram_flat = get_flat_feat(example.mean_cos_dist_2gram)
        mean_leve_dist_2gram_flat = get_flat_feat(example.mean_leve_dist_2gram)
        mean_cos_dist_3gram_flat = get_flat_feat(example.mean_cos_dist_3gram)
        mean_leve_dist_3gram_flat = get_flat_feat(example.mean_leve_dist_3gram)
        mean_cos_dist_4gram_flat = get_flat_feat(example.mean_cos_dist_4gram)
        mean_leve_dist_4gram_flat = get_flat_feat(example.mean_leve_dist_4gram)
        mean_cos_dist_5gram_flat = get_flat_feat(example.mean_cos_dist_5gram)
        mean_leve_dist_5gram_flat = get_flat_feat(example.mean_leve_dist_5gram)
        char_entity_flat = get_flat_feat(example.char_entity)

        for (doc_span_index, doc_span) in enumerate(doc_spans):
            tokens = []
            token_to_orig_map = {}
            token_is_max_context = {}
            segment_ids = []

            # 额外特征, 这里会将question和doc拼接
            char_pos = []
            char_kw = []
            char_in_que = []
            fuzzy_matching_ratio = []
            fuzzy_matching_partial_ratio = []
            fuzzy_matching_token_sort_ratio = []
            fuzzy_matching_token_set_ratio = []
            word_match_share = []
            f1_score = []
            mean_cos_dist_2gram = []
            mean_leve_dist_2gram = []
            mean_cos_dist_3gram = []
            mean_leve_dist_3gram = []
            mean_cos_dist_4gram = []
            mean_leve_dist_4gram = []
            mean_cos_dist_5gram = []
            mean_leve_dist_5gram = []
            char_entity = []

            # p_mask: mask with 1 for token than cannot be in the answer (0 for token which can be in an answer)
            # Original TF implem also keep the classification token (set to 0) (not sure why...)
            p_mask = []

            # CLS token at the beginning, 本次比赛默认是在最前面, 不然有些代码需要调整
            if not cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = 0

                char_pos.append(POS2ID['blank'])
                char_kw.append(0)
                char_in_que.append(0)
                fuzzy_matching_ratio.append(0.0)
                fuzzy_matching_partial_ratio.append(0.0)
                fuzzy_matching_token_sort_ratio.append(0.0)
                fuzzy_matching_token_set_ratio.append(0.0)
                word_match_share.append(0.0)
                f1_score.append(0.0)
                mean_cos_dist_2gram.append(0.0)
                mean_leve_dist_2gram.append(1.0)
                mean_cos_dist_3gram.append(0.0)
                mean_leve_dist_3gram.append(1.0)
                mean_cos_dist_4gram.append(0.0)
                mean_leve_dist_4gram.append(1.0)
                mean_cos_dist_5gram.append(0.0)
                mean_leve_dist_5gram.append(1.0)
                char_entity.append(NER2ID[''])

            # Query
            for item_ in example.ques_char_pos:
                char_pos += [item_[0]] * item_[1]
            for item_ in example.ques_char_kw:
                char_kw += [item_[0]] * item_[1]
            for item_ in example.ques_char_entity:
                char_entity += [item_[0]] * item_[1]
            char_pos = char_pos[:len(query_tokens) + 1]
            char_kw = char_kw[:len(query_tokens) + 1]
            char_entity = char_entity[:len(query_tokens) + 1]

            for token in query_tokens:
                tokens.append(token)
                segment_ids.append(sequence_a_segment_id)
                p_mask.append(1)

                char_in_que.append(0)
                fuzzy_matching_ratio.append(0.0)
                fuzzy_matching_partial_ratio.append(0.0)
                fuzzy_matching_token_sort_ratio.append(0.0)
                fuzzy_matching_token_set_ratio.append(0.0)
                word_match_share.append(0.0)
                f1_score.append(0.0)
                mean_cos_dist_2gram.append(0.0)
                mean_leve_dist_2gram.append(1.0)
                mean_cos_dist_3gram.append(0.0)
                mean_leve_dist_3gram.append(1.0)
                mean_cos_dist_4gram.append(0.0)
                mean_leve_dist_4gram.append(1.0)
                mean_cos_dist_5gram.append(0.0)
                mean_leve_dist_5gram.append(1.0)

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_a_segment_id)
            p_mask.append(1)

            char_pos.append(POS2ID['blank'])
            char_kw.append(0)
            char_in_que.append(0)
            fuzzy_matching_ratio.append(0.0)
            fuzzy_matching_partial_ratio.append(0.0)
            fuzzy_matching_token_sort_ratio.append(0.0)
            fuzzy_matching_token_set_ratio.append(0.0)
            word_match_share.append(0.0)
            f1_score.append(0.0)
            mean_cos_dist_2gram.append(0.0)
            mean_leve_dist_2gram.append(1.0)
            mean_cos_dist_3gram.append(0.0)
            mean_leve_dist_3gram.append(1.0)
            mean_cos_dist_4gram.append(0.0)
            mean_leve_dist_4gram.append(1.0)
            mean_cos_dist_5gram.append(0.0)
            mean_leve_dist_5gram.append(1.0)
            char_entity.append(NER2ID[''])

            # Paragraph
            for i in range(doc_span.length):
                split_token_index = doc_span.start + i
                token_to_orig_map[len(tokens)] = tok_to_orig_index[split_token_index]

                is_max_context = _check_is_max_context(doc_spans, doc_span_index,
                                                       split_token_index)
                token_is_max_context[len(tokens)] = is_max_context
                tokens.append(all_doc_tokens[split_token_index])
                segment_ids.append(sequence_b_segment_id)
                p_mask.append(0)

                char_pos.append(char_pos_flat[split_token_index])
                char_kw.append(char_kw_flat[split_token_index])
                char_in_que.append(char_in_que_flat[split_token_index])
                fuzzy_matching_ratio.append(fuzzy_matching_ratio_flat[split_token_index])
                fuzzy_matching_partial_ratio.append(fuzzy_matching_partial_ratio_flat[split_token_index])
                fuzzy_matching_token_sort_ratio.append(fuzzy_matching_token_sort_ratio_flat[split_token_index])
                fuzzy_matching_token_set_ratio.append(fuzzy_matching_token_set_ratio_flat[split_token_index])
                word_match_share.append(word_match_share_flat[split_token_index])
                f1_score.append(f1_score_flat[split_token_index])
                mean_cos_dist_2gram.append(mean_cos_dist_2gram_flat[split_token_index])
                mean_leve_dist_2gram.append(mean_leve_dist_2gram_flat[split_token_index])
                mean_cos_dist_3gram.append(mean_cos_dist_3gram_flat[split_token_index])
                mean_leve_dist_3gram.append(mean_leve_dist_3gram_flat[split_token_index])
                mean_cos_dist_4gram.append(mean_cos_dist_4gram_flat[split_token_index])
                mean_leve_dist_4gram.append(mean_leve_dist_4gram_flat[split_token_index])
                mean_cos_dist_5gram.append(mean_cos_dist_5gram_flat[split_token_index])
                mean_leve_dist_5gram.append(mean_leve_dist_5gram_flat[split_token_index])
                char_entity.append(char_entity_flat[split_token_index])
            paragraph_len = doc_span.length

            # SEP token
            tokens.append(sep_token)
            segment_ids.append(sequence_b_segment_id)
            p_mask.append(1)

            char_pos.append(POS2ID['blank'])
            char_kw.append(0)
            char_in_que.append(0)
            fuzzy_matching_ratio.append(0.0)
            fuzzy_matching_partial_ratio.append(0.0)
            fuzzy_matching_token_sort_ratio.append(0.0)
            fuzzy_matching_token_set_ratio.append(0.0)
            word_match_share.append(0.0)
            f1_score.append(0.0)
            mean_cos_dist_2gram.append(0.0)
            mean_leve_dist_2gram.append(1.0)
            mean_cos_dist_3gram.append(0.0)
            mean_leve_dist_3gram.append(1.0)
            mean_cos_dist_4gram.append(0.0)
            mean_leve_dist_4gram.append(1.0)
            mean_cos_dist_5gram.append(0.0)
            mean_leve_dist_5gram.append(1.0)
            char_entity.append(NER2ID[''])

            # CLS token at the end
            if cls_token_at_end:
                tokens.append(cls_token)
                segment_ids.append(cls_token_segment_id)
                p_mask.append(0)
                cls_index = len(tokens) - 1  # Index of classification token

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            input_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            # Zero-pad up to the sequence length.
            while len(input_ids) < max_seq_length:
                input_ids.append(pad_token)
                input_mask.append(0 if mask_padding_with_zero else 1)
                segment_ids.append(pad_token_segment_id)
                p_mask.append(1)

                char_pos.append(POS2ID['blank'])
                char_kw.append(0)
                char_in_que.append(0)
                fuzzy_matching_ratio.append(0.0)
                fuzzy_matching_partial_ratio.append(0.0)
                fuzzy_matching_token_sort_ratio.append(0.0)
                fuzzy_matching_token_set_ratio.append(0.0)
                word_match_share.append(0.0)
                f1_score.append(0.0)
                mean_cos_dist_2gram.append(0.0)
                mean_leve_dist_2gram.append(1.0)
                mean_cos_dist_3gram.append(0.0)
                mean_leve_dist_3gram.append(1.0)
                mean_cos_dist_4gram.append(0.0)
                mean_leve_dist_4gram.append(1.0)
                mean_cos_dist_5gram.append(0.0)
                mean_leve_dist_5gram.append(1.0)
                char_entity.append(NER2ID[''])

            assert len(input_ids) == max_seq_length
            assert len(input_mask) == max_seq_length
            assert len(segment_ids) == max_seq_length

            assert len(char_pos) == max_seq_length
            assert len(char_kw) == max_seq_length
            assert len(char_in_que) == max_seq_length
            assert len(fuzzy_matching_ratio) == max_seq_length
            assert len(fuzzy_matching_partial_ratio) == max_seq_length
            assert len(fuzzy_matching_token_sort_ratio) == max_seq_length
            assert len(fuzzy_matching_token_set_ratio) == max_seq_length
            assert len(word_match_share) == max_seq_length
            assert len(f1_score) == max_seq_length
            assert len(mean_cos_dist_2gram) == max_seq_length
            assert len(mean_leve_dist_2gram) == max_seq_length
            assert len(mean_cos_dist_3gram) == max_seq_length
            assert len(mean_leve_dist_3gram) == max_seq_length
            assert len(mean_cos_dist_4gram) == max_seq_length
            assert len(mean_leve_dist_4gram) == max_seq_length
            assert len(mean_cos_dist_5gram) == max_seq_length
            assert len(mean_leve_dist_5gram) == max_seq_length

            span_is_impossible = example.is_impossible
            start_position = None
            end_position = None
            if is_training and not span_is_impossible:
                # For training, if our document chunk does not contain an annotation
                # we throw it out, since there is nothing to predict.
                doc_start = doc_span.start
                doc_end = doc_span.start + doc_span.length - 1
                out_of_span = False
                if not (tok_start_position >= doc_start and
                        tok_end_position <= doc_end):
                    out_of_span = True
                if out_of_span:
                    start_position = 0
                    end_position = 0
                    span_is_impossible = True
                else:
                    doc_offset = len(query_tokens) + 2
                    start_position = tok_start_position - doc_start + doc_offset
                    end_position = tok_end_position - doc_start + doc_offset

            # TODO：不包含答案，负样本采样策略：
            # 1. 设置负样本丢弃概率阈值，random < 阈值，丢弃该负样本
            # 2. 根据该 question id 中可能包含多个答案的 doc_span，不同 qid 的doc_span也不同，按照包含答案和不包含答案的比例进行负样本采样
            if is_training and span_is_impossible:
                if random.random() < args.train_neg_sample_ratio:
                    continue

                start_position = cls_index
                end_position = cls_index

            if example_index < 0:
                logger.info("*** Example ***")
                logger.info("unique_id: %s" % (unique_id))
                logger.info("qas_id: %s" % (example.qas_id))
                logger.info("doc_position: %s" % (example.doc_position))
                logger.info("example_index: %s" % (example_index))
                logger.info("doc_span_index: %s" % (doc_span_index))
                logger.info("tokens: %s" % " ".join(tokens))
                logger.info("token_to_orig_map: %s" % " ".join([
                    "%d:%d" % (x, y) for (x, y) in token_to_orig_map.items()]))
                logger.info("token_is_max_context: %s" % " ".join([
                    "%d:%s" % (x, y) for (x, y) in token_is_max_context.items()
                ]))
                logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
                logger.info(
                    "input_mask: %s" % " ".join([str(x) for x in input_mask]))
                logger.info(
                    "segment_ids: %s" % " ".join([str(x) for x in segment_ids]))
                if is_training and span_is_impossible:
                    logger.info("impossible example")
                if is_training and not span_is_impossible:
                    answer_text = " ".join(tokens[start_position:(end_position + 1)])
                    logger.info("start_position: %d" % (start_position))
                    logger.info("end_position: %d" % (end_position))
                    logger.info("answer: %s" % (answer_text))

            features.append({
                'unique_id': unique_id,
                'example_index': example_index,
                'doc_span_index': doc_span_index,
                'tokens': tokens,
                'token_to_orig_map': token_to_orig_map,
                'token_is_max_context': token_is_max_context,
                'input_ids': input_ids,
                'input_mask': input_mask,
                'segment_ids': segment_ids,
                'p_mask': p_mask,
                'paragraph_len': paragraph_len,
                'start_position': start_position,
                'end_position': end_position,
                'is_impossible': span_is_impossible,
                'doc_position': example.doc_position,
                'char_pos': char_pos,
                'char_kw': char_kw,
                'char_in_que': char_in_que,
                'fuzzy_matching_ratio': fuzzy_matching_ratio,
                'fuzzy_matching_partial_ratio': fuzzy_matching_partial_ratio,
                'fuzzy_matching_token_sort_ratio': fuzzy_matching_token_sort_ratio,
                'fuzzy_matching_token_set_ratio': fuzzy_matching_token_set_ratio,
                'word_match_share': word_match_share,
                'f1_score': f1_score,
                'mean_cos_dist_2gram': mean_cos_dist_2gram,
                'mean_leve_dist_2gram': mean_leve_dist_2gram,
                'mean_cos_dist_3gram': mean_cos_dist_3gram,
                'mean_leve_dist_3gram': mean_leve_dist_3gram,
                'mean_cos_dist_4gram': mean_cos_dist_4gram,
                'mean_leve_dist_4gram': mean_leve_dist_4gram,
                'mean_cos_dist_5gram': mean_cos_dist_5gram,
                'mean_leve_dist_5gram': mean_leve_dist_5gram,
                'char_entity': char_entity})
            unique_id += 1

    # 打印未识别或者被跳过的token
    logger.warning('######print unk and skipped tokens with their counts######')
    logger.warning('the unk tokens is : {}'.format(unk_tokens_dict))
    logger.warning('the skipped tokens is : {}'.format(skipped_tokens_dict))
    # unk_file = '{}_unk_tokens_dict.txt'.format('train' if is_training else 'predict')
    # skipped_file = '{}_skipped_tokens_dict.txt'.format('train' if is_training else 'predict')
    # with open(unk_file, 'w') as fout:
    #     for token, cnt in unk_tokens_dict.items():
    #         fout.write('{}  {}\n'.format(token, cnt))
    # with open(skipped_file, 'w') as fout:
    #     for token, cnt in skipped_tokens_dict.items():
    #         fout.write('{}  {}\n'.format(token, cnt))

    return features


def _improve_answer_span(doc_tokens, input_start, input_end, tokenizer,
                         orig_answer_text):
    """Returns tokenized answer spans that better match the annotated answer."""

    # The SQuAD annotations are character based. We first project them to
    # whitespace-tokenized words. But then after WordPiece tokenization, we can
    # often find a "better match". For example:
    #
    #   Question: What year was John Smith born?
    #   Context: The leader was John Smith (1895-1943).
    #   Answer: 1895
    #
    # The original whitespace-tokenized answer will be "(1895-1943).". However
    # after tokenization, our tokens will be "( 1895 - 1943 ) .". So we can match
    # the exact answer, 1895.
    #
    # However, this is not always possible. Consider the following:
    #
    #   Question: What country is the top exporter of electornics?
    #   Context: The Japanese electronics industry is the lagest in the world.
    #   Answer: Japan
    #
    # In this case, the annotator chose "Japan" as a character sub-span of
    # the word "Japanese". Since our WordPiece tokenizer does not split
    # "Japanese", we just use "Japanese" as the annotation. This is fairly rare
    # in SQuAD, but does happen.
    tok_answer_text = " ".join(tokenizer.tokenize(orig_answer_text))

    for new_start in range(input_start, input_end + 1):
        for new_end in range(input_end, new_start - 1, -1):
            text_span = " ".join(doc_tokens[new_start:(new_end + 1)])
            if text_span == tok_answer_text:
                return (new_start, new_end)

    return (input_start, input_end)


def _check_is_max_context(doc_spans, cur_span_index, position):
    """Check if this is the 'max context' doc span for the token."""

    # Because of the sliding window approach taken to scoring documents, a single
    # token can appear in multiple documents. E.g.
    #  Doc: the man went to the store and bought a gallon of milk
    #  Span A: the man went to the
    #  Span B: to the store and bought
    #  Span C: and bought a gallon of
    #  ...
    #
    # Now the word 'bought' will have two scores from spans B and C. We only
    # want to consider the score with "maximum context", which we define as
    # the *minimum* of its left and right context (the *sum* of left and
    # right context will always be the same, of course).
    #
    # In the example the maximum context for 'bought' would be span C since
    # it has 1 left context and 3 right context, while span B has 4 left context
    # and 0 right context.
    best_score = None
    best_span_index = None
    for (span_index, doc_span) in enumerate(doc_spans):
        end = doc_span.start + doc_span.length - 1
        if position < doc_span.start:
            continue
        if position > end:
            continue
        num_left_context = position - doc_span.start
        num_right_context = end - position
        score = min(num_left_context, num_right_context) + 0.01 * doc_span.length
        if best_score is None or score > best_score:
            best_score = score
            best_span_index = span_index

    return cur_span_index == best_span_index


RawResult = collections.namedtuple("RawResult",
                                   ["unique_id", "start_logits", "end_logits"])

def write_predictions(task_name, all_examples, all_features, all_results, n_best_size,
                      max_answer_length, do_lower_case, output_prediction_file,
                      output_nbest_file, output_null_log_odds_file, verbose_logging,
                      version_2_with_negative, null_score_diff_threshold):
    """Write final predictions to the json file and log-odds of null if needed."""
    logger.info("Writing predictions to: %s" % (output_prediction_file))
    logger.info("Writing nbest to: %s" % (output_nbest_file))

    example_index_to_features = collections.defaultdict(list)
    for feature in all_features:
        example_index_to_features[feature['example_index']].append(feature)

    unique_id_to_result = {}
    for result in all_results:
        unique_id_to_result[result.unique_id] = result

    _PrelimPrediction = collections.namedtuple(  # pylint: disable=invalid-name
        "PrelimPrediction",
        ["feature_index", "start_index", "end_index", "start_logit", "end_logit"])

    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict()

    for (example_index, example) in enumerate(all_examples):
        features = example_index_to_features[example_index]

        prelim_predictions = []
        # keep track of the minimum score of null start+end of position 0
        score_null = 1000000  # large and positive
        min_null_feature_index = 0  # the paragraph slice with min null score
        null_start_logit = 0  # the start logit at the slice with min null score
        null_end_logit = 0  # the end logit at the slice with min null score
        for (feature_index, feature) in enumerate(features):
            result = unique_id_to_result[feature['unique_id']]
            start_indexes = _get_best_indexes(result.start_logits, n_best_size)
            end_indexes = _get_best_indexes(result.end_logits, n_best_size)
            # if we could have irrelevant answers, get the min score of irrelevant
            if version_2_with_negative:
                feature_null_score = result.start_logits[0] + result.end_logits[0]
                if feature_null_score < score_null:
                    score_null = feature_null_score
                    min_null_feature_index = feature_index
                    null_start_logit = result.start_logits[0]
                    null_end_logit = result.end_logits[0]
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # We could hypothetically create invalid predictions, e.g., predict
                    # that the start of the span is in the question. We throw out all
                    # invalid predictions.
                    if start_index >= len(feature['tokens']):
                        continue
                    if end_index >= len(feature['tokens']):
                        continue
                    if start_index not in feature['token_to_orig_map']:
                        continue
                    if end_index not in feature['token_to_orig_map']:
                        continue
                    if not feature['token_is_max_context'].get(start_index, False):
                        continue
                    if end_index < start_index:
                        continue
                    length = end_index - start_index + 1
                    if length > max_answer_length:
                        continue
                    prelim_predictions.append(
                        _PrelimPrediction(
                            feature_index=feature_index,
                            start_index=start_index,
                            end_index=end_index,
                            start_logit=result.start_logits[start_index],
                            end_logit=result.end_logits[end_index]))
        if version_2_with_negative:
            prelim_predictions.append(
                _PrelimPrediction(
                    feature_index=min_null_feature_index,
                    start_index=0,
                    end_index=0,
                    start_logit=null_start_logit,
                    end_logit=null_end_logit))
        prelim_predictions = sorted(
            prelim_predictions,
            key=lambda x: (x.start_logit + x.end_logit),
            reverse=True)

        _NbestPrediction = collections.namedtuple(  # pylint: disable=invalid-name
            "NbestPrediction", ["text", "start_logit", "end_logit"])

        seen_predictions = {}
        nbest = []
        for pred in prelim_predictions:
            if len(nbest) >= n_best_size:
                break
            feature = features[pred.feature_index]
            if pred.start_index > 0:  # this is a non-null prediction
                # tok_tokens = feature['tokens'][pred.start_index:(pred.end_index + 1)]
                orig_doc_start = feature['token_to_orig_map'][pred.start_index]
                orig_doc_end = feature['token_to_orig_map'][pred.end_index]
                orig_tokens = example.doc_tokens[orig_doc_start:(orig_doc_end + 1)]
                # tok_text = " ".join(tok_tokens)

                # De-tokenize WordPieces that have been split off.
                # tok_text = tok_text.replace(" ##", "")
                # tok_text = tok_text.replace("##", "")

                # Clean whitespace, 这里空格暂时不处理
                # tok_text = tok_text.strip()
                # tok_text = " ".join(tok_text.split())
                # orig_text = " ".join(orig_tokens)
                orig_text = "".join(orig_tokens)

                # 暂时不用get_final_text
                # final_text = get_final_text(tok_text, orig_text, do_lower_case, verbose_logging)
                final_text = orig_text
                if final_text in seen_predictions:
                    continue

                seen_predictions[final_text] = True
            else:
                final_text = ""
                seen_predictions[final_text] = True

            nbest.append(
                _NbestPrediction(
                    text=final_text,
                    start_logit=pred.start_logit,
                    end_logit=pred.end_logit))
        # if we didn't include the empty option in the n-best, include it
        if version_2_with_negative:
            if "" not in seen_predictions:
                nbest.append(
                    _NbestPrediction(
                        text="",
                        start_logit=null_start_logit,
                        end_logit=null_end_logit))

            # In very rare edge cases we could only have single null prediction.
            # So we just create a nonce prediction in this case to avoid failure.
            if len(nbest)==1:
                nbest.insert(0,
                    _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        # In very rare edge cases we could have no valid predictions. So we
        # just create a nonce prediction in this case to avoid failure.
        if not nbest:
            nbest.append(
                _NbestPrediction(text="empty", start_logit=0.0, end_logit=0.0))

        assert len(nbest) >= 1

        total_scores = []
        best_non_null_entry = None
        for entry in nbest:
            total_scores.append(entry.start_logit + entry.end_logit)
            if not best_non_null_entry:
                if entry.text:
                    best_non_null_entry = entry

        probs = _compute_softmax(total_scores)

        nbest_json = []
        for (i, entry) in enumerate(nbest):
            output = collections.OrderedDict()
            output["text"] = entry.text
            output["probability"] = probs[i]
            output["start_logit"] = entry.start_logit
            output["end_logit"] = entry.end_logit
            nbest_json.append(output)

        assert len(nbest_json) >= 1

        if not version_2_with_negative:
            all_predictions[example.qas_id] = nbest_json[0]["text"]
        else:
            score_diff = score_null - best_non_null_entry.start_logit - (
                best_non_null_entry.end_logit)
            scores_diff_json[example.qas_id] = score_diff

            if task_name == ANSWER_MRC:
                # 始终取example中最大的非空答案
                non_null_prob = best_non_null_entry.start_logit + best_non_null_entry.end_logit
                all_predictions[example.qas_id] = [best_non_null_entry.text, non_null_prob]
            else:
                # predict "" iff the null score - the score of best non-null > threshold
                if score_diff > null_score_diff_threshold:
                    all_predictions[example.qas_id] = ["", score_null]
                else:
                    all_predictions[example.qas_id] = best_non_null_entry.text
                    non_null_prob = best_non_null_entry.start_logit + best_non_null_entry.end_logit
                    all_predictions[example.qas_id] = [best_non_null_entry.text, non_null_prob]

        all_nbest_json[example.qas_id] = nbest_json

    if task_name == ANSWER_MRC:
        all_samples = collections.defaultdict(list)
        need_skippeed_list = [',', '，', '.', '。']
        for qas_id, nbest_json in all_nbest_json.items():
            text = ''
            prob = 0.0
            logit = 0.0
            for entry in nbest_json:
                if entry['text'].strip() and entry['text'] not in need_skippeed_list:
                    text = entry['text'].strip()
                    logit = entry['start_logit'] + entry['end_logit']
                    prob = entry['probability']
                    break
            all_samples[qas_id.split('##')[0]].append([text, logit, prob])
        all_predictions = {}
        for question_id, sample in all_samples.items():
            sample = sorted(sample, key=lambda x: x[1], reverse=True)
            all_predictions[question_id] = sample[0][0]
            # 简单的多答案选择模块
            # if sample[1][0] != '' and sample[1][2] > 0.1:
            #     # 有可能具有多答案
            #     if sample[1][0] in sample[0][0] or sample[0][0] in sample[1][0]:
            #         continue
            #     ans1 = normalize([sample[0][0]])
            #     ans2 = normalize([sample[1][0]])
            #     bleu_rouge = compute_bleu_rouge({'item': ans1}, {'item': ans2})
            #     if bleu_rouge['Bleu-4'] > 0.5 or bleu_rouge['Rouge-L'] > 0.5:
            #         continue
            #     logger.warning('{} have multi-ans, take care of it'.format(question_id))
            #     all_predictions[question_id] = all_predictions[question_id] + '#' + sample[1][0]
    else:
        # 将多个example合成一个sample, 根据阈值决定是否有答案
        all_samples = collections.defaultdict(list)
        for qas_id, prediction in all_predictions.items():
            all_samples[qas_id.split('##')[0]].append(prediction)
        all_predictions = {}
        for question_id, sample in all_samples.items():
            sample = sorted(sample, key=lambda x: x[1], reverse=True)
            for ex in sample:
                if ex[0] != "":
                    all_predictions[question_id] = ex[0]
                    break
            if question_id not in all_predictions:
                all_predictions[question_id] = ""

    # # 将多个example合成一个sample
    # all_samples = collections.defaultdict(list)
    # for qas_id, prediction in all_predictions.items():
    #     all_samples[qas_id.split('##')[0]].append(prediction)
    # all_predictions = {}
    # for question_id, sample in all_samples.items():
    #     sample = sorted(sample, key=lambda x: x[1], reverse=True)
    #     all_predictions[question_id] = sample[0][0]

    with open(output_prediction_file, "w") as writer:
        writer.write(json.dumps(all_predictions, indent=4, ensure_ascii=False) + "\n")

    with open(output_nbest_file, "w") as writer:
        writer.write(json.dumps(all_nbest_json, indent=4, ensure_ascii=False) + "\n")

    if version_2_with_negative:
        with open(output_null_log_odds_file, "w") as writer:
            writer.write(json.dumps(scores_diff_json, indent=4, ensure_ascii=False) + "\n")

    return all_predictions


# For XLNet (and XLM which uses the same head)
RawResultExtended = collections.namedtuple("RawResultExtended",
    ["unique_id", "start_top_log_probs", "start_top_index",
     "end_top_log_probs", "end_top_index", "cls_logits"])


def write_predictions_extended(all_examples, all_features, all_results, n_best_size,
                                max_answer_length, output_prediction_file,
                                output_nbest_file,
                                output_null_log_odds_file, orig_data_file,
                                start_n_top, end_n_top, version_2_with_negative,
                                tokenizer, verbose_logging):
    """ XLNet write prediction logic (more complex than Bert's).
        Write final predictions to the json file and log-odds of null if needed.

        Requires utils_squad_evaluate.py
    """
    out_eval = {}

    return out_eval


def get_final_text(pred_text, orig_text, do_lower_case, verbose_logging=False):
    """Project the tokenized prediction back to the original text."""

    # When we created the data, we kept track of the alignment between original
    # (whitespace tokenized) tokens and our WordPiece tokenized tokens. So
    # now `orig_text` contains the span of our original text corresponding to the
    # span that we predicted.
    #
    # However, `orig_text` may contain extra characters that we don't want in
    # our prediction.
    #
    # For example, let's say:
    #   pred_text = steve smith
    #   orig_text = Steve Smith's
    #
    # We don't want to return `orig_text` because it contains the extra "'s".
    #
    # We don't want to return `pred_text` because it's already been normalized
    # (the SQuAD eval script also does punctuation stripping/lower casing but
    # our tokenizer does additional normalization like stripping accent
    # characters).
    #
    # What we really want to return is "Steve Smith".
    #
    # Therefore, we have to apply a semi-complicated alignment heuristic between
    # `pred_text` and `orig_text` to get a character-to-character alignment. This
    # can fail in certain cases in which case we just return `orig_text`.

    def _strip_spaces(text):
        ns_chars = []
        ns_to_s_map = collections.OrderedDict()
        for (i, c) in enumerate(text):
            if c == " ":
                continue
            ns_to_s_map[len(ns_chars)] = i
            ns_chars.append(c)
        ns_text = "".join(ns_chars)
        return (ns_text, ns_to_s_map)

    # We first tokenize `orig_text`, strip whitespace from the result
    # and `pred_text`, and check if they are the same length. If they are
    # NOT the same length, the heuristic has failed. If they are the same
    # length, we assume the characters are one-to-one aligned.
    tokenizer = BasicTokenizer(do_lower_case=do_lower_case)

    tok_text = " ".join(tokenizer.tokenize(orig_text))

    start_position = tok_text.find(pred_text)
    if start_position == -1:
        if verbose_logging:
            logger.info(
                "Unable to find text: '%s' in '%s'" % (pred_text, orig_text))
        return orig_text
    end_position = start_position + len(pred_text) - 1

    (orig_ns_text, orig_ns_to_s_map) = _strip_spaces(orig_text)
    (tok_ns_text, tok_ns_to_s_map) = _strip_spaces(tok_text)

    if len(orig_ns_text) != len(tok_ns_text):
        if verbose_logging:
            logger.info("Length not equal after stripping spaces: '%s' vs '%s'",
                        orig_ns_text, tok_ns_text)
        return orig_text

    # We then project the characters in `pred_text` back to `orig_text` using
    # the character-to-character alignment.
    tok_s_to_ns_map = {}
    for (i, tok_index) in tok_ns_to_s_map.items():
        tok_s_to_ns_map[tok_index] = i

    orig_start_position = None
    if start_position in tok_s_to_ns_map:
        ns_start_position = tok_s_to_ns_map[start_position]
        if ns_start_position in orig_ns_to_s_map:
            orig_start_position = orig_ns_to_s_map[ns_start_position]

    if orig_start_position is None:
        if verbose_logging:
            logger.info("Couldn't map start position")
        return orig_text

    orig_end_position = None
    if end_position in tok_s_to_ns_map:
        ns_end_position = tok_s_to_ns_map[end_position]
        if ns_end_position in orig_ns_to_s_map:
            orig_end_position = orig_ns_to_s_map[ns_end_position]

    if orig_end_position is None:
        if verbose_logging:
            logger.info("Couldn't map end position")
        return orig_text

    output_text = orig_text[orig_start_position:(orig_end_position + 1)]
    return output_text


def _get_best_indexes(logits, n_best_size):
    """Get the n-best logits from a list."""
    index_and_score = sorted(enumerate(logits), key=lambda x: x[1], reverse=True)

    best_indexes = []
    for i in range(len(index_and_score)):
        if i >= n_best_size:
            break
        best_indexes.append(index_and_score[i][0])
    return best_indexes


def _compute_softmax(scores):
    """Compute softmax probability over raw logits."""
    if not scores:
        return []

    max_score = None
    for score in scores:
        if max_score is None or score > max_score:
            max_score = score

    exp_scores = []
    total_sum = 0.0
    for score in scores:
        x = math.exp(score - max_score)
        exp_scores.append(x)
        total_sum += x

    probs = []
    for score in exp_scores:
        probs.append(score / total_sum)
    return probs
