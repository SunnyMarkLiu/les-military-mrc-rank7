#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/12 21:28
"""
import sys
sys.path.append('../')
import re
import json
import time
from utils.baidu_translate import BaiduTranslator

translator = BaiduTranslator()

ans_pattern = re.compile(r'@content\d@')


def find_answer_in_docid(answer):
    docs = ans_pattern.findall(answer)
    return list(set([int(doc[-2:-1]) for doc in docs]))


def back_translate(text):
    """
    zh -> en -> zh
    """
    result = translator.translate(text, src='zh', dst='fra')
    result = result['trans_result']['data'][0]['dst']

    back_content = translator.translate(result, src='fra', dst='zh')
    back_content = back_content['trans_result']['data'][0]['dst']
    time.sleep(1)

    return back_content

if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())

        sample['question'] = back_translate(sample['question'])
        time.sleep(1)

        back_trans_answer = ''
        ans_in_docids = find_answer_in_docid(sample['answer'])
        for ans_in_docid in ans_in_docids:
            flag_str = '@content{}@'.format(ans_in_docid)
            answer_strs = sample['answer'].split('@content{}@'.format(ans_in_docid))
            for answer_str in answer_strs:
                answer_str = answer_str.strip()  # important
                # @content1@ 包裹的实际答案文本
                if answer_str != '' and '@content' not in answer_str:
                    answer_str = answer_str.replace('content{}@'.format(ans_in_docid), '')
                    back_trans_answer += (flag_str + back_translate(answer_str) + flag_str)
        sample['answer'] = back_trans_answer
        time.sleep(1)

        back_supporting_paragraph = ''
        sup_para_in_docids = find_answer_in_docid(sample['supporting_paragraph'])
        for docid in sup_para_in_docids:
            flag_str = '@content{}@'.format(docid)
            sup_para_strs = sample['answer'].split('@content{}@'.format(docid))
            for sup_para_str in sup_para_strs:
                sup_para_str = sup_para_str.strip()  # important
                # @content1@ 包裹的实际答案文本
                if sup_para_str != '' and '@content' not in sup_para_str:
                    sup_para_str = sup_para_str.replace('content{}@'.format(docid), '')
                    back_supporting_paragraph += (flag_str + back_translate(sup_para_str) + flag_str)
        sample['supporting_paragraph'] = back_supporting_paragraph
        time.sleep(1)

        for doc in sample['documents']:
            doc['content'] = back_translate(doc['content'])
            time.sleep(1)

        del sample['fake_answers']
        del sample['ceil_rougel']
        print(json.dumps(sample, ensure_ascii=False))
