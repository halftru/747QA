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
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from simple_github_implementation.model import QAModel
from scipy.stats import rankdata

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

sentence_length = 200
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


def pad(data, length):
    return pad_sequences(data, maxlen=length, padding='post', truncating='post', value=0)


def process_data(answersy, sentence_length, d):
    indices = d['good'] + d['bad']
    answers = pad([answersy[i] for i in indices], sentence_length)
    question = pad([d['question']] * len(indices), sentence_length)
    return indices, answers, question


testf = pd.read_csv("./data/trec/test.csv")
trainf = pd.read_csv("./data/trec/train-all.csv")
valf = pd.read_csv("./data/trec/dev.csv")
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
tokenizer.fit_on_texts(train2)

emb_glove = glove(tokenizer)
emb_glove = np.load('./data/trec_emb')

print(emb_glove)


input_file = open('./data/trec/train-all.csv', 'r', encoding='utf-8')

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

questions = tokenizer.texts_to_sequences(questions)
good_answers = tokenizer.texts_to_sequences(good_answers)
bad_answers = tokenizer.texts_to_sequences(bad_answers)

questions = pad(questions, sentence_length)
good_answers = pad(good_answers, sentence_length)
bad_answers = pad(bad_answers, sentence_length)

print(questions)
input_file.close()

qa_model = QAModel()
train_model, predict_model = qa_model.get_lstm_cnn_model(emb_glove)
epo = 1


callbacks = [EarlyStopping(monitor='val_loss', patience=20),
             ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

# train the model
Y = np.zeros(shape=(len(questions),))
train_model.fit([questions, good_answers, bad_answers], Y, epochs=epo, batch_size=64, validation_split=0.1,
                verbose=1, callbacks=callbacks)

# save the trained model
# train_model.save_weights('model/train_weights_epoch_' + str(epo) + '.h5', overwrite=True)
predict_model.save_weights('model/predict_weights_epoch_' + str(epo) + '.h5', overwrite=True)

exit()
# load the evaluation data
data = []
random.shuffle(data)

# load weights from trained model
model_filenames = ['model/predict_weights_epoch_' + str(epo) + '.h5']

for model_name in model_filenames:
    predict_model.load_weights(model_name)

    c = 0
    c1 = 0
    for i, d in enumerate(data):
        if i % 100 == 0:
            print(i, len(data))

        # pad the data and get it in desired format
        indices, answers, question = process_data(d)

        # get the similarity score
        sims = predict_model.predict([question, answers])

        n_good = len(d['good'])
        max_r = np.argmax(sims)
        max_n = np.argmax(sims[:n_good])
        r = rankdata(sims, method='max')
        c += 1 if max_r == max_n else 0
        c1 += 1 / float(r[max_r] - r[max_n] + 1)

    print(f'Results for: model: {model_name}')
    print("top1", c / float(len(data)))
    print("MRR", c1 / float(len(data)))