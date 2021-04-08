import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from model import QAModel
from data import QAData, Vocabulary
import pickle
import random
from scipy.stats import rankdata
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.callbacks import EarlyStopping, ModelCheckpoint

physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

def main(mode='test', question=None, answers=None):
    """
    This function is used to train, predict or test

    Args:
        mode (str): train/preddict/test
        question (str): this contains the question
        answers (list): this contains list of answers in string format

    Returns:
        index (integer): index of the most likely answer
    """

    # get the train and predict model model
    vocabulary = Vocabulary("./data/vocab_all.txt")
    embedding_file = "./data/word2vec_100_dim.embeddings"
    qa_model = QAModel()
    train_model, predict_model = qa_model.get_lstm_cnn_model(embedding_file, len(vocabulary))

    epochs = 50
    if mode == 'train':
        # load training data
        qa_data = QAData()
        questions, good_answers, bad_answers = qa_data.get_training_data()

        callbacks = [EarlyStopping(monitor='val_loss', patience=100),
                     ModelCheckpoint(filepath='model/best_model.h5', monitor='val_loss', save_best_only=True)]

        # train the model
        Y = np.zeros(shape=(questions.shape[0],))
        history = train_model.fit(
            [questions, good_answers, bad_answers],
            Y,
            epochs=epochs,
            batch_size=64,
            validation_split=0.1,
            verbose=1,
            callbacks=callbacks
        )

        # list all data in history
        print(history.history.keys())
        # summarize history for loss
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        fig = plt.gcf()
        fig.savefig('history_train_weights_epoch_' + str(epochs) + '.png')
        # plt.show()

        # save the trained model
        train_model.save_weights('model/train_weights_epoch_' + str(epochs) + '.h5', overwrite=True)
        predict_model.save_weights('model/predict_weights_epoch_' + str(epochs) + '.h5', overwrite=True)

    elif mode == 'predict':
        # load the evaluation data
        data = pickle.load(open("./data/dev.pkl",'rb'))
        random.shuffle(data)

        # load weights from trained model
        qa_data = QAData()
        model_filenames = ['model/best_model.h5', 'model/predict_weights_epoch_' + str(epochs) + '.h5']

        for model_name in model_filenames:
            predict_model.load_weights(model_name)

            c = 0
            c1 = 0
            for i, d in enumerate(data):
                if i%100 == 0:
                    print(i, len(data))

                # pad the data and get it in desired format
                indices, answers, question = qa_data.process_data(d)

                # get the similarity score
                sims = predict_model.predict([question, answers])

                n_good = len(d['good'])
                max_r = np.argmax(sims)
                max_n = np.argmax(sims[:n_good])
                r = rankdata(sims, method='max')
                c += 1 if max_r == max_n else 0
                c1 += 1 / float(r[max_r] - r[max_n] + 1)

            precision = c / float(len(data))
            mrr = c1 / float(len(data))
            print(f'Results for: model: {model_name}')
            print("Precision", precision)
            print("MRR", mrr)
    elif mode == 'test':
        # question and answers come from params
        qa_data = QAData()
        answers, question = qa_data.process_test_data(question, answers)

        # load weights from the trained model
        predict_model.load_weights('model/predict_weights_epoch_' + str(epochs) + '.h5')

        # get similarity score
        sims = predict_model.predict([question, answers])
        max_r = np.argmax(sims)
        return max_r

if __name__ == "__main__":
    main(mode='train')
    main(mode='predict')

def test(question, answers):
    return main(mode='test', question=question, answers=answers)
