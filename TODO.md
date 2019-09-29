# TODO LIST
## 数据预处理
- text_cleaning.py 中计算 match score 是否考虑 para ngram ==> para ngram 效果较好，考虑了前后上下文，score=0的与supporting para重合较少，区分度高
- 统计段落的长度分布，以及答案是否在长段落的情况，决定是否需要对长段落进行拆分
    - 段落筛选难以保证答案覆盖率
- 存在重复问题

## 特征工程
- https://github.com/ShusenTang/BDC2019
- https://github.com/srtianxia/BDC2019_Rank2

## bert-baseline相关
- 训练被截断的例子的处理
- 分词器可以优化, 参照CMRC2018的自定义分词器; 另外CMRC2018还增加了input_span_mask字段,可以试一试
- 对OOV的词进行处理
- 一些小细节用TODO标签定位了
- 对于被截断的训练集是否需要丢弃

## 多任务
根据 start end logits 的最大值定位的答案所在的句子，作为模型选出来的 “support para”，然后和question进行类似 NLI （或者简单点的attention），得到模型筛选答案的依据“support para”，是否和问题问的相关的

- entity 规则
# 可能存在推理entity的关键词
首款|批|架|次|艘，唯一，第(一|1)(名|位|批|架|款|代|种|艘)，下一代，新一代，开山鼻祖，
其，取代了，它，多个问题的规则，该，成为，“，
最大|小|晚|早|多|少|强|顽强|好|差|低|高|先进，
外号，号称，戏称，绰号，宣称，被称作，被称为，取代的，替代的，
同一类型，所属国，的(国家|导弹|航母|直升机|手枪|飞机|运输机|军舰|反导系统|输送车|核动力航母|潜艇),
