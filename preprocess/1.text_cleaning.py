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

# remove space
spaces = {'\x10', '\x7f', '\x9d', '\xad', '\x0a', '\xa0', '\x0d', '\u001d', '\u0007', '\u001f', '\u000f',
          '\u0001', '\u0002', '\u0003', '\u0004', '\u0005', '\u0006', '\u0007', '\u0008', '\u0009',
          '\u0010', '\u0011', '\u0012', '\u0013', '\u0014', '\u0015', '\u0016', '\u0017', '\u0018',
          '\f', '\n', '\r', '\t', '\v', '&#160;', '&nbsp;', '\\uDDEF', '\\uDDEE', '\\uDDE8', '\\uDDF3', '\\u0001',
          '\\uD83C', '\\uDDFA', '\\uD83C', '\\uDDF8', '\\uDDEA', '\\uDDF7', '\\uDDF5', '\\uDDF3',
          '\u200b', '\u200e', '\u202a', '\u202c', '\ufeff', '\uf0d8', '\u2061', '\u1680', '\u180e',
          '\u2000', '\u2001', '\u2002', '\u2003', '\u2004', '\u2005', '\u2006', '\u2007', '\u2008',
          '\u2009', '\u200a', '\u2028', '\u2029', '\u202f', '\u205f', '\u3000'}


def _remove_space(text):
    for space in spaces:
        text = text.replace(space, '')
    text = re.sub('\s+', ' ', text)
    return text


def clean_document(document):
    title = document['title']
    paragraphs = document['paragraphs']

    document['paragraphs'] = [_remove_space(para) for para in document['paragraphs']]
    return document


def clean_sample(sample):
    documents = [clean_document(document) for document in sample['documents']]
    sample['documents'] = documents
    return True


if __name__ == '__main__':
    for line in sys.stdin:
        if not line.startswith('{'):
            continue

        sample = json.loads(line.strip())
        if clean_sample(sample):
            # _nlp_text_analyse(sample)
            print(json.dumps(sample, ensure_ascii=False))
