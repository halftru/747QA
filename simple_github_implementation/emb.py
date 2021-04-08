from __future__ import print_function
import pickle

from keras_preprocessing.sequence import pad_sequences
from keras_preprocessing.text import Tokenizer
import numpy as np


def revert(vocab, indices):
    return [vocab.get(i, 'X') for i in indices]


vocab = pickle.load(open("./data/vocabulary.pkl", 'rb'))
max_features = len(vocab)

answers = pickle.load(open("./data/answers.pkl", 'rb'))
sentences = [revert(vocab, txt) for txt in answers.values()]
sentences += [revert(vocab, q['question']) for q in pickle.load(open("./data/train.pkl", 'rb'))]
tokenizer = Tokenizer(num_words=200)
tokenizer.fit_on_texts(sentences)
sentences = tokenizer.texts_to_sequences(sentences)
sentences = pad_sequences(sentences, 100)
embeddings_index = {}

# ignore errors and specify encoding to avoid some errors
f = open('./data/glove.840B.300d.txt', errors='ignore', encoding='utf-8')
for line in f:
    values = line.split()
    word = ''.join(values[:-100])
    co = np.asarray(values[-100:], dtype='float32')
    embeddings_index[word] = co
f.close()
thing = np.stack(embeddings_index.values())

word_index = tokenizer.word_index
embedding_matrix_1 = np.random.normal(thing.mean(), thing.std(),
                                      (min(max_features, len(word_index)), thing.shape[1]))

for word, i in word_index.items():
    if max_features > i:
        break
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix_1[i] = embedding_vector
print("done with glove")
print(embedding_matrix_1.shape)
np.save("glove_100dim.embeddings", embedding_matrix_1)
