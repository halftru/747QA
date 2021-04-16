import csv
from collections import defaultdict

import numpy as np
import pandas as pd
import time
import random

from gensim.models import KeyedVectors
from keras.preprocessing.sequence import pad_sequences
from keras import layers, Sequential
# from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer
from keras.wrappers.scikit_learn import KerasClassifier
import tensorflow as tf
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint

from model import QAModel
from scipy.stats import rankdata
import time

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)


def word2(tok):
    embeddings_index = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin.gz', binary=True)
    word_index = tok.word_index
    embedding_matrix_2 = np.zeros(shape=(len(word_index)+1, 300), dtype='float32')
    for i, word in word_index.items():
        if word in embeddings_index:
            embedding_vector = embeddings_index.get_vector(word)
            embedding_matrix_2[i] = embedding_vector
    return embedding_matrix_2


def fast(tok):
    embeddings_index = KeyedVectors.load_word2vec_format('./data/wiki-news-300d-1M.vec', binary=True)
    word_index = tok.word_index
    embedding_matrix_2 = np.zeros(shape=(len(word_index)+1, 300), dtype='float32')
    for i, word in word_index.items():
        if word in embeddings_index:
            embedding_vector = embeddings_index.get_vector(word)
            embedding_matrix_2[i] = embedding_vector
    return embedding_matrix_2


def embedding(tok, emb_file):
    embeddings_index = {}

    # ignore stuff that causes errors and specify encoding to avoid some errors
    f = open(emb_file, errors='ignore', encoding='utf-8')
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

    return embedding_matrix_1


def pad(data, length):
    return pad_sequences(data, maxlen=length, padding='post', truncating='post', value=0)


def preprocess_train_file(filename, tokenizer, sentence_length=200):

    input_file = open(filename, 'r', encoding='utf-8')

    reader = csv.reader(input_file, delimiter=',')

    positives = []
    negatives = defaultdict(list)
    answers = set()
    for i, (question, label, answer) in enumerate(reader):
        if i!=0:
            if label == "1":
                positives.append((question, answer))
            else:
                negatives[question].append(answer)
            answers.add(answer)
    input_file.close()
    answers = list(answers)

    questions = []
    good_answers = []
    bad_answers = []
    for question, positive in positives:
        if len(negatives[question]) > 0:
            negative = negatives[question].pop()
        else:
            negative = random.choice(answers)
            while negative == positive:
                negative = random.choice(answers)

        questions.append(question)
        good_answers.append(positive)
        bad_answers.append(negative)

    questions_tokenized = tokenizer.texts_to_sequences(questions)
    questions = pad(questions_tokenized, sentence_length)
    good_tokenized = tokenizer.texts_to_sequences(good_answers)
    good_answers = pad(good_tokenized, sentence_length)
    bad_tokenized = tokenizer.texts_to_sequences(bad_answers)
    bad_answers = pad(bad_tokenized, sentence_length)
    return questions, good_answers, bad_answers


def preprocess_test_file(filename, tokenizer, sentence_length=200):
    input_file = open(filename, 'r', encoding='utf-8')
    reader = csv.reader(input_file, delimiter=',')

    nested = defaultdict(lambda: defaultdict(list)) # {'(question)':{'good': [answers], 'bad': [answers], 'question': [question repeated]}}
    # questions = {'': defaultdict(list)}
    # answers = defaultdict(list) # {'': []}
    for i, (question, label, answer) in enumerate(reader):
        if i!=0:
            good_bad = 'good' if label == "1" else 'bad'

            nested[question][good_bad].append(answer)
            nested[question]['question'].append(question)

    input_file.close()

    for i, (question, answers_dict) in enumerate(nested.items()):

        answer_good_tokenized = tokenizer.texts_to_sequences(answers_dict['good'])
        answer_good_cleaned = pad(answer_good_tokenized, sentence_length)
        answers_dict['good'] = answer_good_cleaned
        answer_bad_tokenized = tokenizer.texts_to_sequences(answers_dict['bad'])
        answer_bad_cleaned = pad(answer_bad_tokenized, sentence_length)
        answers_dict['bad'] = answer_bad_cleaned
        answer_question_tokenized = tokenizer.texts_to_sequences(answers_dict['question'])
        answer_question_cleaned = pad(answer_question_tokenized, sentence_length)
        answers_dict['question'] = answer_question_cleaned
        
    return nested


def train(tokenizer, sentence_length, epochs=1, batch_size=64, save=True):

    questions, good_answers, bad_answers = preprocess_train_file('./data/trec/train-all.csv', tokenizer, sentence_length)
    val_questions, val_good_answers, val_bad_answers = preprocess_train_file('./data/trec/dev.csv', tokenizer, sentence_length)

    callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

    # # train the model
    Y = np.zeros(shape=(len(questions),))
    train_model.fit([questions, good_answers, bad_answers], Y, epochs=epochs, batch_size=batch_size, 
                    validation_data=[val_questions, val_good_answers, val_bad_answers],
                    verbose=1, callbacks=callbacks)

    # save the trained model
    if save:
        predict_model.save_weights('model/trec/weights_epoch_' + str(epochs) + '.h5', overwrite=True)


def test(tokenizer, sentence_length, epochs=1):
    # load the evaluation data
    test_data_dict_list = preprocess_test_file('./data/trec/test.csv', tokenizer, sentence_length)

    # load weights from trained model
    model_filenames = ['model/trec/weights_epoch_' + str(epochs) + '.h5']

    for model_name in model_filenames:
        predict_model.load_weights(model_name)

        c = 0
        c1 = 0
        for i, (question, answers_dict) in enumerate(test_data_dict_list.items()):
            n_good = len(answers_dict['good'])
            if n_good == 0:
                print('No good answers for question: {question}')
                continue
            
            answers = np.concatenate((answers_dict['good'], answers_dict['bad']))    # saves to good
            
            # get the similarity score
            sims = predict_model.predict([answers_dict['question'], answers])

            max_r = np.argmax(sims)
            max_n = np.argmax(sims[:n_good])
            r = rankdata(sims, method='max')
            c += 1 if max_r == max_n else 0
            c1 += 1 / float(r[max_r] - r[max_n] + 1)

        print(f'Results for: model: {model_name}')
        print("top1", c / float(len(test_data_dict_list)))
        print("MRR", c1 / float(len(test_data_dict_list)))
        print(f'# Questions: {len(test_data_dict_list)}')


if __name__ == "__main__":
    sentence_length = 200
    epochs = 1
    batch_size=64

    # Read in data files
    testf = pd.read_csv("./data/trec/test.csv")
    trainf = pd.read_csv("./data/trec/train-all.csv")
    valf = pd.read_csv("./data/trec/dev.csv")
    
    train_all_text = np.hstack(trainf['qtext'] + trainf['atext'])

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(train_all_text)

    # Load Embeddings
    emb_glove = embedding(tokenizer, './data/glove.6B.300d.txt')
    emb_word2 = word2(tokenizer)
    emb_fast = fast(tokenizer)
    exit()

    qa_model = QAModel()
    train_model, predict_model = qa_model.get_lstm_cnn_model(emb_glove)

    train(tokenizer, sentence_length, epochs=epochs, batch_size=batch_size, save=True)
    test(tokenizer, sentence_length, epochs=epochs)

