# les-military-mrc

## bert-baseline相关
### 说明
- *bert-baseline主要基于[pytorch-transformers](https://github.com/huggingface/pytorch-transformers)搭建*
- pytorch-transformers库目前一直在修复bug, 直接pip安装的是1.1.0版本, 也许有问题~

### 已完成
- copy了截止2019/9/2号master分支上最新的`run_squad.py`、`utils_squad.py`和`utils_squad_evaluate.py`文件
- 针对多分档任务进行了一些改写工作, 初步在dureader数据集上跑通
- copy了dureader的eval_metric进行复用
