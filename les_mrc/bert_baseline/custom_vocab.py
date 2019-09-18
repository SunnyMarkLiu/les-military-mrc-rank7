#!/usr/bin/python
# _*_ coding: utf-8 _*_

"""

@author: Qing Liu, sunnymarkliu@163.com
@github: https://github.com/sunnymarkLiu
@time  : 2019/9/18 20:36
"""
import collections
import numpy as np
from tqdm import tqdm
import logging
logger = logging.getLogger(__name__)


class CustomVocab(object):

    def __init__(self, vocab_file, do_lower_case):
        """
        注意，此自定义词典和 bert 的词典共用一个 vocab.txt，而且采用list方式进行 tokenize，因此输入 id 可复用 bert 的 indice,
        进而基于此 CustomVocab 编码输出可以和 bert 输出进行拼接

        :param vocab_file:
        :param do_lower_case:
        """
        logger.info('load bert vocab file')
        self.vocab = load_vocab(vocab_file)
        self.ids_to_tokens = collections.OrderedDict([(ids, tok) for tok, ids in self.vocab.items()])
        self.do_lower_case = do_lower_case
        self.unk_token = '[UNK]'
        self.embed_dim = None
        self.embedding_matrix = None


    def _convert_token_to_id(self, token):
        """ Converts a token (str/unicode) in an id using the vocab. """
        return self.vocab.get(token, self.vocab.get(self.unk_token))

    def _convert_id_to_token(self, index):
        """Converts an index (integer) in a token (string/unicode) using the vocab."""
        return self.ids_to_tokens.get(index, self.unk_token)

    @property
    def vocab_size(self):
        return len(self.vocab)

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.
        Args:
            embeddings_file: A file containing pretrained word embeddings.
        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings into a dictionnary.
        embeddings_index, emb_mean, emb_std, self.embed_dim = load_pretrained_embed(embeddings_file)
        logger.info('pretrained embeddings mean: {}, std: {}, calc from top 1000 words'.format(emb_mean, emb_std))

        # build embedding matrix
        self.embedding_matrix = np.random.normal(emb_mean, emb_std, (self.vocab_size, self.embed_dim))

        oov_words = []
        for word, idx in self.vocab.items():
            if self.do_lower_case:
                word = word.lower()

            if word in embeddings_index:
                self.embedding_matrix[idx] = embeddings_index[word]
            else:
                oov_words.append(word)

        with open('oov_chars.txt', 'w') as oov_writer:
            oov_writer.writelines(['\n'.join(oov_words)])


def load_vocab(vocab_file):
    """Loads a vocabulary file into a dictionary."""
    vocab = collections.OrderedDict()
    with open(vocab_file, "r", encoding="utf-8") as reader:
        tokens = reader.readlines()
    for index, token in enumerate(tokens):
        token = token.rstrip('\n')
        vocab[token] = index
    return vocab

def load_pretrained_embed(filepath):
    """
    load pretrained embeddings
    """
    embeddings_index = {}

    with open(filepath, 'r', encoding='utf-8') as f:
        vocab_size, embed_dim = map(int, f.readline().strip().split(" "))

        for _ in tqdm(range(vocab_size)):
            lists = f.readline().rstrip().split(" ")
            word = lists[0]
            vector = np.asarray(list(map(float, lists[1:])), dtype='float16')
            embeddings_index[word] = vector

    sample_embs = np.stack(list(embeddings_index.values())[:1000])
    emb_mean, emb_std = sample_embs.mean(), sample_embs.std()

    return embeddings_index, emb_mean, emb_std, embed_dim
