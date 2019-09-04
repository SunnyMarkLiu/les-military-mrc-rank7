#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/3 21:59
"""
import re
import sys
import json
import collections
from collections import Counter


# --------------- remove space ---------------
spaces = {'\x10', '\x7f', '\x9d', '\xad', '\x0a', '\xa0', '\x0d', '\u001d', '\u0007', '\u001f', '\u000f',
          '\u0001', '\u0002', '\u0003', '\u0004', '\u0005', '\u0006', '\u0007', '\u0008', '\u0009',
          '\u0010', '\u0011', '\u0012', '\u0013', '\u0014', '\u0015', '\u0016', '\u0017', '\u0018',
          '\f', '\n', '\r', '\t', '\v', '&#160;', '&nbsp;', '\\uDDEF', '\\uDDEE', '\\uDDE8', '\\uDDF3', '\\u0001',
          '\\uD83C', '\\uDDFA', '\\uD83C', '\\uDDF8', '\\uDDEA', '\\uDDF7', '\\uDDF5', '\\uDDF3',
          '\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u1680', '\u180e',
          '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008',
          '\u2009', '\u200a', '\u2028', '\u2029', '\u202f', '\u205f', '\u3000'}


def remove_unicode_space(text):
    for space in spaces:
        text = text.replace(space, '')
    text = re.sub('\s+', ' ', text)
    return text


# --------------- remove url ---------------
URL_REGEX1 = r"""(?i)\b((?:https?:(?:/{1,3}|[a-z0-9%])|[a-z0-9.\-]+[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)/)(?:[^\s()<>{}\[\]]+|\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\))+(?:\([^\s()]*?\([^\s()]+\)[^\s()]*?\)|\([^\s]+?\)|[^\s`!()\[\]{};:\'\".,<>?«»“”‘’])|(?:(?<!@)[a-z0-9]+(?:[.\-][a-z0-9]+)*[.](?:com|net|org|edu|gov|mil|aero|asia|biz|cat|coop|info|int|jobs|mobi|museum|name|post|pro|tel|travel|xxx|ac|ad|ae|af|ag|ai|al|am|an|ao|aq|ar|as|at|au|aw|ax|az|ba|bb|bd|be|bf|bg|bh|bi|bj|bm|bn|bo|br|bs|bt|bv|bw|by|bz|ca|cc|cd|cf|cg|ch|ci|ck|cl|cm|cn|co|cr|cs|cu|cv|cx|cy|cz|dd|de|dj|dk|dm|do|dz|ec|ee|eg|eh|er|es|et|eu|fi|fj|fk|fm|fo|fr|ga|gb|gd|ge|gf|gg|gh|gi|gl|gm|gn|gp|gq|gr|gs|gt|gu|gw|gy|hk|hm|hn|hr|ht|hu|id|ie|il|im|in|io|iq|ir|is|it|je|jm|jo|jp|ke|kg|kh|ki|km|kn|kp|kr|kw|ky|kz|la|lb|lc|li|lk|lr|ls|lt|lu|lv|ly|ma|mc|md|me|mg|mh|mk|ml|mm|mn|mo|mp|mq|mr|ms|mt|mu|mv|mw|mx|my|mz|na|nc|ne|nf|ng|ni|nl|no|np|nr|nu|nz|om|pa|pe|pf|pg|ph|pk|pl|pm|pn|pr|ps|pt|pw|py|qa|re|ro|rs|ru|rw|sa|sb|sc|sd|se|sg|sh|si|sj|Ja|sk|sl|sm|sn|so|sr|ss|st|su|sv|sx|sy|sz|tc|td|tf|tg|th|tj|tk|tl|tm|tn|to|tp|tr|tt|tv|tw|tz|ua|ug|uk|us|uy|uz|va|vc|ve|vg|vi|vn|vu|wf|ws|ye|yt|yu|za|zm|zw)\b/?(?!@)))"""
URL_REGEX2 = r'[^\u4e00-\u9fa5|[$-_@.&+]]*(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'


def remove_url_links(text):
    text = re.sub(URL_REGEX1, '', text)
    text = re.sub(URL_REGEX2, '', text)
    return text


# --------------- remove html tag ---------------
html_pattern = re.compile(r'<.*?>')
zhPattern = re.compile(u'[\u4e00-\u9fa5]+')


def remove_html_tag(text):
    text = re.sub('<{2,}', '', text)  # <<<
    # <> 中包含中文的
    text = re.sub('<；；', '', text)
    text = re.sub('([\u4e00-\u9fa5]+)>', '\g<1>', text)
    text = re.sub('<([\u4e00-\u9fa5]+)>', '\g<1>', text)
    tags = html_pattern.findall(text)

    new_tags = set()
    for tag in tags:
        match = zhPattern.search(tag)
        if not match:
            new_tags.add(tag)

    for html_tag in new_tags:
        text = text.replace(html_tag, '')

    return text


# --------------- remove duplacte chars ---------------
def clean_duplacte_chars(text):
    """
    去除很多重复的词和标点符号
    """
    reg = r'([^0-9IX]+)(\1){2,}'
    for i in range(6):
        temp = text
        text = re.sub(reg, lambda m: m.group(1), text)
        if len(text) == len(temp):
            break
    return text


remove_regx_map = collections.OrderedDict({
    r'\s+': ' ',
    r'<(\d+)>': '\g<1>',
    r'\^[A-Z]': '',  # '^G', '^H', '^E'去除
    r'(\!|\"|\#|\$|\%|\&|\'|\(|\)|\*|\+|\,|\-|\.|\/|\:|\;|\<|\=|\>|\?|\@|\[|\\|\]|\^|\_|\`|\{|\||\}|\~)\1{1,}': '\g<1>',
    r'("""|＃|＄|％|＆|＇|（|）|＊|＋|，|－|／|：|；|＜|＝|＞|＠|［|＼|］|＾|＿|｀|｛|｜|｝|～|｟|｠|｢|｣|､|　|、|〃|〈|〉|《|》|'
    r'「|」|『|』|【|】|〔|〕|〖|〗|〘|〙|〚|〛|〜|〝|〞|〟|〰|〾|〿|–|—|‘|’|‛|“|”|„|‟|…|‧|﹏|﹑|﹔|·|！|？|｡|。)\1{1,}': '\g<1>',
})


def remove_by_regex(text):
    text = text.strip()
    for rgx in remove_regx_map:
        text = re.sub(rgx, remove_regx_map[rgx], text)
    return text


def clean_text(text, is_supporting_paragraph=False):
    text = remove_unicode_space(text)
    text = remove_html_tag(text)

    if not is_supporting_paragraph:
        text = remove_url_links(text)

    text = clean_duplacte_chars(text)
    text = remove_by_regex(text)
    return text


def clean_sample(sample):
    sample['question'] = clean_text(sample['question'])
    sample['keyword'] = clean_text(sample['keyword'])
    if 'supporting_paragraph' in sample:
        sample['supporting_paragraph'] = clean_text(sample['supporting_paragraph'], is_supporting_paragraph=True)

    for document in sample['documents']:
        document['paragraphs'] = [clean_text(para) for para in document['paragraphs']]
        document['title'] = clean_text(document['title'])

        # 去除空段落和重复段落
        new_paras = []
        for para in document['paragraphs']:
            if para != '' and para not in new_paras:
                new_paras.append(para)
        document['paragraphs'] = new_paras

# -------------------- 计算 paragraph 与question，supporting_paragraph 与 paragraph 的匹配得分 --------------------

def precision_recall_f1(prediction, ground_truth):
    """
    This function calculates and returns the precision, recall and f1-score
    Args:
        prediction: prediction string or list to be matched
        ground_truth: golden string or list reference
    Returns:
        floats of (p, r, f1)
    Raises:
        None
    """
    if not isinstance(prediction, list):
        prediction_tokens = list(prediction)
    else:
        prediction_tokens = prediction
    if not isinstance(ground_truth, list):
        ground_truth_tokens = list(ground_truth)
    else:
        ground_truth_tokens = ground_truth

    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0, 0, 0
    p = 1.0 * num_same / len(prediction_tokens)
    r = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * p * r) / (p + r)
    return p, r, f1


# def ques_para_match_score(question, para):
#     """
#     计算问题和段落的匹配得分
#
#     Args:
#         question: str
#         para: str
#     """
#     p, r, f1 = precision_recall_f1(para, question)
#     match_score = p, r, f1
#     return match_score


content_pattern = re.compile(r'@content\d@')
def find_support_para_in_docid(support_para):
    docs = content_pattern.findall(support_para)
    return list(set([int(doc[-2:-1]) for doc in docs]))

def calc_match_scores_paras(sample):
    """
    计算 paragraph 与question，supporting_paragraph 与 paragraph 的匹配得分
    用于后期的 paragraph selection
    """
    for doc in sample['documents']:
        para_match_precisions = []
        para_match_recalls = []
        para_match_f1s = []
        for pid, para in enumerate(doc['paragraphs']):
            pre_para = doc['paragraphs'][pid - 1] if pid > 0 else ''
            ngramed_para = pre_para + para

            match_scores = precision_recall_f1(ngramed_para, sample['question'] + sample['keyword'])
            para_match_precisions.append(match_scores[0])
            para_match_recalls.append(match_scores[1])
            para_match_f1s.append(match_scores[2])
        doc['para_match_precisions'] = para_match_precisions
        doc['para_match_recalls'] = para_match_recalls
        doc['para_match_f1s'] = para_match_f1s

    # 找到答案所支撑的 docid
    if 'supporting_paragraph' not in sample:
        return

    supported_doc_ids = find_support_para_in_docid(sample['supporting_paragraph'])
    for sid in supported_doc_ids:
        doc = sample['documents'][sid - 1]

        # 该 doc 中支撑答案的段落 id
        supported_para_ids = []

        sents = sample['supporting_paragraph'].split('@content{}@'.format(sid))
        for supported_sent in sents:
            if supported_sent != '' and '@content' not in supported_sent:
                # 找到 supported_sent 所在的 para id
                max_recall = -1
                supported_para_id = None
                for pid, para in enumerate(doc['paragraphs']):
                    pre_para = doc['paragraphs'][pid - 1] if pid > 0 else ''
                    ngramed_para = pre_para + para

                    recall = precision_recall_f1(ngramed_para, supported_sent)[1]
                    if recall > max_recall:
                        supported_para_id = pid
                        max_recall = recall

                supported_para_ids.append(supported_para_id)

        supported_para_ids = sorted(list(set(supported_para_ids)))
        doc['supported_para_ids'] = supported_para_ids

if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        clean_sample(sample)
        calc_match_scores_paras(sample)
        print(json.dumps(sample, ensure_ascii=False))
