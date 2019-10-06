#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
分词和关键词、POS 标注

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/6 09:19
"""
import os
import sys

sys.path.append('../')
import sys
import json
from utils.jieba_util import WordSegmentPOSKeywordExtractor

jieba_extractor = WordSegmentPOSKeywordExtractor()
print('jieba prepared.')
# pip install foolnltk
from fool.predictor import Predictor
from fool.lexical import LexicalAnalyzer
from zipfile import ZipFile
import tensorflow as tf

def _load_map_file(path, char_map_name, id_map_name):
    with ZipFile(path) as myzip:
        with myzip.open('all_map.json') as myfile:
            content = myfile.readline()
            content = content.decode()
            data = json.loads(content)
            return data.get(char_map_name), data.get(id_map_name)

def load_graph(path):
    with tf.gfile.GFile(path, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name="prefix")
    return graph

class NERPredictor(Predictor):
    def __init__(self, model_file, char_to_id, id_to_tag):
        super().__init__(model_file, char_to_id, id_to_tag)
        self.char_to_id = char_to_id
        self.id_to_tag = {int(k):v for k,v in id_to_tag.items()}
        self.graph = load_graph(model_file)

        self.input_x = self.graph.get_tensor_by_name("prefix/char_inputs:0")
        self.lengths = self.graph.get_tensor_by_name("prefix/lengths:0")
        self.dropout = self.graph.get_tensor_by_name("prefix/dropout:0")
        self.logits = self.graph.get_tensor_by_name("prefix/project/logits:0")
        self.trans = self.graph.get_tensor_by_name("prefix/crf_loss/transitions:0")

        sess_config = tf.ConfigProto()
        sess_config.gpu_options.allow_growth = True
        self.sess = tf.Session(config=sess_config, graph=self.graph)
        self.sess.as_default()
        self.num_class = len(self.id_to_tag)


class NERAnalyzer(LexicalAnalyzer):

    def _load_model(self, model_namel, word_map_name, tag_name):
        seg_model_path = os.path.join(self.data_path, model_namel)
        char_to_id, id_to_seg = _load_map_file(self.map_file_path, word_map_name, tag_name)
        return NERPredictor(seg_model_path, char_to_id, id_to_seg)

    def analysis(self, text_list):
        ners = self.ner(text_list)
        return ners

NER_ANALYZER = NERAnalyzer()

def fetch_ner(text, entities):
    # 处理 char 的 entity 边界
    char_i = 0
    entity_i = 0
    char_entity = []
    while char_i < len(text):
        if entity_i == len(entities):
            char_entity.append('')
            char_i += 1
            continue
        if char_i < entities[entity_i][0]:  # 非实体词的 char
            char_entity.append('')
            char_i += 1
        elif entities[entity_i][0] <= char_i < entities[entity_i][0] + len(entities[entity_i][3]):
            char_entity.append(entities[entity_i][2])
            char_i += 1
        else:
            entity_i += 1

    new_char_entity = []
    for entity in char_entity:
        if entity == 'time':
            entity = 'T'
        elif entity == 'location':
            entity = 'L'
        elif entity == 'org':
            entity = 'O'
        elif entity == 'job':
            entity = 'J'
        elif entity == 'person':
            entity = 'P'
        elif entity == 'company':
            entity = 'C'
        new_char_entity.append(entity)
    return new_char_entity

def extract_ner_features(sample):
    """
    将 question 和 content 进行拼接，加速 ner 抽取
    """
    text_list = [sample['question']] + [doc['content'] for doc in sample['documents']]
    ners = NER_ANALYZER.analysis(text_list)

    que_ners = ners[0]
    doc_ners = ners[1:]

    sample['ques_char_entity'] = fetch_ner(sample['question'], que_ners)
    for doc_i, doc in enumerate(sample['documents']):
        doc['char_entity'] = fetch_ner(doc['content'], doc_ners[doc_i])

def extract_text_features(text, question_sapns):
    seg, term_pos, term_kw = jieba_extractor.extract_sentence(text, 0.4)
    term_in_que = [int(token in question_sapns) for token in seg]
    # 处理 term 的 entity 边界问题
    # 处理 char 的 pos，keyword，in_que
    char_pos, char_kw, char_in_que = [], [], []
    char_pointer = 0
    for term_i, term in enumerate(seg):
        char_pos.extend([term_pos[term_i]] * len(term))
        char_kw.extend([term_kw[term_i]] * len(term))
        char_in_que.extend([term_in_que[term_i]] * len(term))
        char_pointer += len(term)

    return char_pos, char_kw, char_in_que


def text_analysis(sample):
    """
    中文分词，关键词提取，POS标注
    """
    # question
    char_pos, char_kw, char_in_que = extract_text_features(text=sample['question'], question_sapns=set())
    sample['ques_char_pos'] = char_pos
    sample['ques_char_kw'] = char_kw
    sample['ques_char_in_que'] = char_in_que

    question_sapns = set(sample['question'])
    for doc in sample['documents']:
        char_pos, char_kw, char_in_que = extract_text_features(text=doc['content'], question_sapns=question_sapns)
        doc['char_pos'] = char_pos
        doc['char_kw'] = char_kw
        doc['char_in_que'] = char_in_que

    # 抽取 NER 特征
    extract_ner_features(sample)

if __name__ == '__main__':
    gpu = sys.argv[1]

    os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    # disable TF debug logs
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # INFO/warning/ERROR/FATAL
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        text_analysis(sample)
        print(json.dumps(sample, ensure_ascii=False))
