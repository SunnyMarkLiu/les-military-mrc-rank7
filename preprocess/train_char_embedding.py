#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/5/27 14:23
"""
import os
import gc
import time
import logging
import multiprocessing
import gensim
from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence

logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

corpus_file = '../input/military_corpus/finetune_lm_corpus.txt'

# 去重后的sentences列表
sentences = set()

with open(corpus_file) as fin:
    for lid, line in enumerate(fin):
        if lid % 1e6 == 0 and lid > 0:
            logger.info(lid / 1e6)
        line = line.strip()
        if line != '':
            sentences.add(line) # query text

logger.info('total sentences: {}'.format(len(sentences)))

# 保存去重后的sentences
with open('../input/embeddings/sentences.txt', 'w') as fout:
    for idx, sentence in enumerate(sentences):
        if idx % 1e6 == 0 and idx > 0: logger.info(idx)
        fout.write(' '.join(list(sentence)) + '\n')

del sentences
gc.collect()

class EpochSaver(gensim.models.callbacks.CallbackAny2Vec):
    """ 用于保存模型, 打印损失函数等等 """

    def __init__(self, savedir, save_name):
        self.savedir = savedir
        self.save_name = save_name
        self.epoch = 0
        self.pre_loss = 0
        self.best_loss = 999999999.9
        self.since = time.time()
        self.should_stop_patience = 1
        self.loss_increase_cnt = 0

    def on_epoch_end(self, model):
        self.epoch += 1
        cum_loss = model.get_latest_training_loss()  # 返回的是从第一个epoch累计的
        logger.info('latest_loss: {}, pre_loss: {}'.format(cum_loss, self.pre_loss))
        epoch_loss = cum_loss - self.pre_loss
        time_taken = time.time() - self.since
        logger.info("==> Epoch %d, loss: %.9f, time: %dmin %ds" %
                    (self.epoch, epoch_loss, time_taken // 60, time_taken % 60))
        if epoch_loss <= self.best_loss:
            self.best_loss = epoch_loss
            logger.info("Better model. Best loss: %.9f" % self.best_loss)
            model.save(os.path.join(self.savedir, self.save_name + '.model'))
            model.wv.save_word2vec_format(os.path.join(self.savedir, self.save_name + '.txt'), binary=False)
            logger.info("Model saved to %s " % self.savedir)
            self.loss_increase_cnt = 0
        else:
            self.loss_increase_cnt += 1
            if self.loss_increase_cnt > self.should_stop_patience:
                logger.info('Best model saved, loss: %.9f' % self.best_loss)
                raise ValueError('should early stop')

        self.pre_loss = cum_loss
        self.since = time.time()

corpus = '../input/embeddings/sentences.txt'
logger.info('train word2vec...')
embed_name = 'cbow_win10_mincnt5_neg5_dim100'

try:
    model = Word2Vec(LineSentence(corpus), sg=0, size=300, window=10, min_count=5, negative=5, iter=20,
                     workers=multiprocessing.cpu_count(), compute_loss=True, alpha=0.025,
                     callbacks=[EpochSaver('../input/embeddings/', embed_name)])
except ValueError:
    logger.info('early stopping!')

logger.info('done.')
