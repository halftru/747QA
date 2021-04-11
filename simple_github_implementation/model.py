from keras import backend as K
from keras.layers import Embedding
from keras.layers import LSTM, Input, merge, Lambda, concatenate, Dot
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import Convolution1D
from keras.models import Model
import numpy as np
from keras.layers.wrappers import Bidirectional
import pandas as pd


class QAModel:
    @staticmethod
    def get_lstm_cnn_model(embedding_file):
        margin = 0.2
        hidden_dim = 141
        sentence_length = 200
        weights = np.load(embedding_file)
        weights = pd.DataFrame(weights)
        weights = weights.loc[(weights!=0).any(axis=1)]
        weights = weights.values
        # initialize the question and answer shapes and datatype
        question = Input(shape=(sentence_length,), dtype='int32', name='question_base')
        answer = Input(shape=(sentence_length,), dtype='int32', name='answer_good_base')
        answer_good = Input(shape=(sentence_length,), dtype='int32', name='answer_good_base')
        answer_bad = Input(shape=(sentence_length,), dtype='int32', name='answer_bad_base')

        # embed the question and answers
        qa_embedding = Embedding(input_dim=weights.shape[0], output_dim=weights.shape[1], weights=[weights])
        question_embedding = qa_embedding(question)
        answer_embedding = qa_embedding(answer)
        # pass the question embedding through bi-lstm
        f_rnn = LSTM(hidden_dim, return_sequences=True)
        b_rnn = LSTM(hidden_dim, return_sequences=True)
        qf_rnn = f_rnn(question_embedding)
        qb_rnn = b_rnn(question_embedding)
        question_pool = concatenate([qf_rnn, qb_rnn],  axis=-1)
        af_rnn = f_rnn(answer_embedding)
        ab_rnn = b_rnn(answer_embedding)
        answer_pool = concatenate([af_rnn, ab_rnn], axis=-1)

        filter_sizes = [1, 2, 3, 5]
        cnns = [Convolution1D(filters=500, kernel_size=ngram_size, activation='tanh', padding='same')
                                   for ngram_size in filter_sizes]

        question_cnn = concatenate([cnn(question_pool) for cnn in cnns])
        answer_cnn = concatenate([cnn(answer_pool) for cnn in cnns])

        # apply max pooling
        maxpool = Lambda(lambda x: K.max(x, axis=1, keepdims=False), output_shape=lambda x: (x[0], x[2]))
        maxpool.supports_masking = True
        question_pool = maxpool(question_cnn)
        answer_pool = maxpool(answer_cnn)

        # get similarity similarity score
        merged_model = Dot(axes=1, normalize=True)([question_pool, answer_pool])
        lstm_convolution_model = Model(inputs=[question, answer], outputs=merged_model, name='lstm_convolution_model')
        good_similarity = lstm_convolution_model([question, answer_good])
        bad_similarity = lstm_convolution_model([question, answer_bad])

        # compute the loss
        loss = Lambda(lambda x: K.relu(x[1] - x[0] + margin))([good_similarity, bad_similarity])

        # return the training and prediction model
        prediction_model = Model(inputs=[question, answer_good], outputs=good_similarity, name='prediction_model')
        prediction_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")
        training_model = Model(inputs=[question, answer_good, answer_bad], outputs=loss, name='training_model')
        training_model.compile(loss=lambda y_true, y_pred: y_pred, optimizer="rmsprop")

        return training_model, prediction_model
