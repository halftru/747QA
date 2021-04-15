import os
import numpy as np
from model import QAModel
import pickle
import random
from scipy.stats import rankdata
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.preprocessing.sequence import pad_sequences

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# let my gpu have memory
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

question = None
answers = None
sentence_length = 200
answersy = pickle.load(open("./data/answers.pkl", 'rb'))
training_set = pickle.load(open("./data/train.pkl", 'rb'))

def pad(data, length):
    return pad_sequences(data, maxlen=length, padding='post', truncating='post', value=0)


def process_data(d):
    indices = d['good'] + d['bad']
    answers = pad([answersy[i] for i in indices], sentence_length)
    question = pad([d['question']] * len(indices), sentence_length)
    return answers, question


def get_training_data():
    questions = []
    good_answers = []
    for j, qa in enumerate(training_set):
        questions.extend([qa['question']] * len(qa['answers']))
        good_answers.extend([answersy[i] for i in qa['answers']])

    # pad the question and answers
    questions = pad(questions, sentence_length)
    good_answers = pad(good_answers, sentence_length)
    bad_answers = pad(random.sample(list(answersy.values()), len(good_answers)), sentence_length)

    return questions, good_answers, bad_answers


def main(mode='test'):
    # get the train and predict model model
    embedding = "fast_100_dim.embeddings"
    embedding = np.load(embedding)
    qa_model = QAModel()
    train_model, predict_model = qa_model.get_lstm_cnn_model(embedding)
    epo = 1
    if mode == 'train':

        # load training data
        questions, good_answers, bad_answers = get_training_data()

        callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]

        # train the model
        Y = np.zeros(shape=(questions.shape[0],))
        train_model.fit([questions, good_answers, bad_answers], Y, epochs=epo, batch_size=128, validation_split=0.1,
                        verbose=1, callbacks=callbacks)

        # save the trained model
        predict_model.save_weights('model/insurance/weights_epoch_' + str(epo) + '.h5', overwrite=True)

    elif mode == 'predict':
        # load the evaluation data
        data = pickle.load(open("./data/dev.pkl", 'rb'))
        random.shuffle(data)

        # load weights from trained model
        model_filenames = ['model/insurance/weights_epoch_' + str(epo) + '.h5']

        for model_name in model_filenames:
            predict_model.load_weights(model_name)

            c = 0
            c1 = 0
            for i, d in enumerate(data):
                if i % 100 == 0:
                    print(i, len(data))

                # pad the data and get it in desired format
                answers, question = process_data(d)

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


if __name__ == "__main__":
    main(mode='train')
    main(mode='predict')
