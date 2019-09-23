"""
第二届Les杯评测函数
"""
import json
import numpy as np
import re
from eval_metric import normalize, compute_rouge


# 考虑了多答案的莱斯杯标准评测函数
def evaluate_on_les_answer(all_predictions, ref_file_path):
    """
    Args:
        all_predictions: dict类型, 其中key代表question_id, value代表预测答案字符串
        ref_file_path: 参考文件路径, 每一行均是json格式
    Return:
        dict类型, rouge-l的分数, e.g. score['Rouge-L'] = 0.80
    """
    ref_dict = {}
    with open(ref_file_path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            answer_list = re.split('@content\d@', sample['answer'])
            answer_list = filter(lambda x: x != '', answer_list)
            answer_list = normalize(answer_list)  # A list of normalized strings
            ref_dict[sample['question_id']] = answer_list
    assert len(all_predictions.keys()) == len(ref_dict.keys())

    sample_rouge_list = []
    for question_id, answer_list in ref_dict.items():
        predictions = normalize(all_predictions[question_id].split('#'))
        item_rouge_list = []
        for pred in predictions:
            one_pred_best_rouge = 0.0
            for ans in answer_list:
                rouge = compute_rouge({'item': [pred]}, {'item': [ans]})['Rouge-L']
                if rouge > one_pred_best_rouge:
                    one_pred_best_rouge = rouge
            item_rouge_list.append(one_pred_best_rouge)
        max_cnt = max(1, max(len(predictions), len(answer_list)))
        item_rouge = sum(item_rouge_list) / max_cnt
        sample_rouge_list.append(item_rouge)
    rouge_score = {'Rouge-L': np.mean(sample_rouge_list)}
    return rouge_score, sample_rouge_list


# 莱斯杯预测bridge entity的评测函数
def evaluate_on_les_bridge_entity(all_predictions, ref_file_path):
    """
    Args:
        all_predictions: dict类型, 其中key代表question_id, value代表预测答案字符串
        ref_file_path: 参考文件路径, 每一行均是json格式
    Return:
        dict类型, rouge-l的分数, e.g. score['Rouge-L'] = 0.80
    """
    ref_dict = {}
    with open(ref_file_path) as fin:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            sample = json.loads(line)
            if sample['bridging_entity'] is None:
                sample['bridging_entity'] = ""
            answer = normalize([sample['bridging_entity']])  # A list of normalized strings
            ref_dict[sample['question_id']] = answer
    assert len(all_predictions.keys()) == len(ref_dict.keys())

    sample_rouge_list = []
    for question_id, answer_list in ref_dict.items():
        predictions = normalize(all_predictions[question_id].split('#'))
        item_rouge_list = []
        for pred in predictions:
            one_pred_best_rouge = 0.0
            for ans in answer_list:
                rouge = compute_rouge({'item': [pred]}, {'item': [ans]})['Rouge-L']
                if rouge > one_pred_best_rouge:
                    one_pred_best_rouge = rouge
            item_rouge_list.append(one_pred_best_rouge)
        max_cnt = max(1, max(len(predictions), len(answer_list)))
        item_rouge = sum(item_rouge_list) / max_cnt
        sample_rouge_list.append(item_rouge)
    rouge_score = {'Rouge-L': np.mean(sample_rouge_list)}
    return rouge_score, sample_rouge_list
