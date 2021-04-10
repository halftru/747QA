from __future__ import print_function
import pickle

from gensim.models import Word2Vec, KeyedVectors, FastText
from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import numpy as np


def rebuild(vocab, indices):
    return [vocab.get(i, 'X') for i in indices]


def glove():
    embeddings_index = {}

    # ignore errors and specify encoding to avoid some errors
    f = open('./data/glove.840B.300d.txt', errors='ignore', encoding='utf-8')
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        co = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = co
    f.close()
    embedding_matrix_1 = np.zeros(shape=(len(vocab)+1, 300), dtype='float32')
    for i, word in vocab.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_1[i] = embedding_vector
    print("done with glove")
    print(embedding_matrix_1.shape)
    np.save("data/pretrained_glove_300_dim.embeddings", embedding_matrix_1)


def word2():
    embeddings_index = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    embedding_matrix_2 = np.zeros(shape=(len(vocab)+1, 300), dtype='float32')
    for i, word in vocab.items():
        if word in embeddings_index:
            embedding_vector = embeddings_index.get_vector(word)
            embedding_matrix_2[i] = embedding_vector
    print("done with word2vec")
    print(embedding_matrix_2.shape)
    np.save("data/pretrained_word2_300_dim.embeddings", embedding_matrix_2)


def word():
    model = Word2Vec(sentences, size=100, min_count=1, iter=10, window=5, sg=1)
    weights = model.wv.vectors
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 1, 100), dtype='float32')

    for i, w in vocab.items():
        if w not in d: continue
        emb[i, :] = weights[d[w], :]

    np.save(open('data/pretrained_word2vec_100_dim.embeddings', 'wb'), emb)


def fast():
    model = FastText(sentences, size=100, min_count=1, iter=10, window=5, sg=1)
    weights = model.wv.vectors
    d = dict([(k, v.index) for k, v in model.wv.vocab.items()])
    emb = np.zeros(shape=(len(vocab) + 1, 100), dtype='float32')

    for i, w in vocab.items():
        if w not in d: continue
        emb[i, :] = weights[d[w], :]

    np.save(open('data/pretrained_fast_100_dim.embeddings', 'wb'), emb)


vocab = pickle.load(open("./data/vocabulary.pkl", 'rb'))
max_features = len(vocab)
print(max_features)
answers = pickle.load(open("./data/answers.pkl", 'rb'))
sentences = [rebuild(vocab, txt) for txt in answers.values()]
sentences += [rebuild(vocab, q['question']) for q in pickle.load(open("./data/train.pkl", 'rb'))]
glove()