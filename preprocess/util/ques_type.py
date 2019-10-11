#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""
问题类型和entity相关规则的特征

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/10/11 16:27
"""
import re

entity_keyword_regs = [
    re.compile('的首[发款级制选战架支飞都舰套批机次要种艘块艇]\S*的'),
    re.compile('((自研|研制|设计|建造|打响|列装|制造|国产化|发射|打造|换装|装备|拥有|生产|研发|定型|使用|失败|成功|开工|服役|现役|退役|举行|实施|被称为|攻占|列装|参加|开发|实现|建造|参加|派出|损失|机型)的第\S*的)'),
    re.compile('最(大|小|晚|早|多|少|强|顽强|好|差|低|高|先进)的\S*的'),
    re.compile('的(国家|导弹|航母|直升机|手枪|飞机|运输机|军舰|反导系统|输送车|核动力航母|潜艇)，'),
    re.compile('(的(新|下)一代)|开山鼻祖|(，[该它])|(成为)'),
    re.compile('鼓吹|外号|号称|戏称|绰号|宣称|被称作|被称为|取代的|替代的'),
    re.compile('？\S+？')
]


def need_bridge_entity_reasoning(question):
    """
    根据问题文本，检测是否包含相关关键词，判断是否需要 bridge entity 推理
    """
    flags = []
    for pattern in entity_keyword_regs:
        flags.append(len(re.findall(pattern, question)) > 0)

    return sum(flags) > 0
