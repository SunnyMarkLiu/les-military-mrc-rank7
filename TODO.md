# TODO LIST
## 数据预处理
- text_cleaning.py 中计算 match score 是否考虑 para ngram ==> para ngram 效果较好，考虑了前后上下文，score=0的与supporting para重合较少，区分度高
- 统计段落的长度分布，以及答案是否在长段落的情况，决定是否需要对长段落进行拆分
    - 段落筛选难以保证答案覆盖率

## bert-baseline相关
- 训练被截断的例子的处理
- 分词器可以优化, 参照CMRC2018的自定义分词器; 另外CMRC2018还增加了input_span_mask字段,可以试一试
- 对OOV的词进行处理
- 一些小细节用TODO标签定位了
- 对于被截断的训练集是否需要丢弃
