import csv
from collections import defaultdict

import numpy as np
import pandas as pd
import time
import random
from keras.preprocessing.sequence import pad_sequences
from keras import layers, Sequential
# from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf


def glove(tok):
    embeddings_index = {}

    # ignore stuff that causes errors and specify encoding to avoid some errors
    f = open('./data/glove.6B.300d.txt', errors='ignore', encoding='utf-8')
    for line in f:
        values = line.split()
        word = ''.join(values[:-300])
        co = np.asarray(values[-300:], dtype='float32')
        embeddings_index[word] = co
    f.close()

    word_index = tok.word_index
    embedding_matrix_1 = np.zeros(shape=(len(tok.word_index) + 1, 300), dtype='float32')

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix_1[i] = embedding_vector
    print("done with glove")
    print(embedding_matrix_1.shape)
    np.save(open('data/trec_emb', 'wb'), embedding_matrix_1)


testf = pd.read_csv("./data/trec/test.csv")
trainf = pd.read_csv("./data/trec/train.csv")
valf = pd.read_csv("./data/trec/dev.csv")
print(testf)
test_label = testf['label']
test2 = np.hstack(testf['qtext'] + testf['atext'])
train_label = trainf['label']
train2 = np.hstack(trainf['qtext'] + trainf['atext'])

tokenizer = Tokenizer()
tokenizer.fit_on_texts(train2)
x_train = tokenizer.texts_to_sequences(train2)
x_test = tokenizer.texts_to_sequences(test2)
x_train = pad_sequences(x_train)
x_test = pad_sequences(x_test)

# emb_glove = glove(tokenizer)
emb_glove = np.load('./data/trec_emb')

print(emb_glove)


input_file = open('./data/trec/train.csv', 'r', encoding='utf-8')

reader = csv.reader(input_file, delimiter=',')

positives = []
negatives = defaultdict(list)
answers = set()
for question, label, answer in reader:
    if label == "1":
        positives.append((question, answer))
    else:
        negatives[question].append(answer)
    answers.add(answer)
print(positives)
answers = list(answers)

questions = []
good_answers = []
bad_answers = []
for question, positive in positives:
    counter = 0
    if len(negatives[question]) > 0:
        negative = negatives[question].pop()
    else:
        negative = random.choice(answers)
        while negative == positive:
            negative = random.choice(answers)

    questions.append(question)
    good_answers.append(positive)
    bad_answers.append(negative)

print(questions)
input_file.close()
