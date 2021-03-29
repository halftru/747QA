import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
from model import QAModel
from data import QAData, Vocabulary
import pickle
import random
from scipy.stats import rankdata
import tensorflow as tf

# let my gpu have memory
physical_devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)

question = None
answers = None


def main(mode='test'):
    # get the train and predict model model
    vocabulary = Vocabulary("./data/vocab_all.txt")
    embedding_file = "./data/word2vec_100_dim.embeddings"
    qa_model = QAModel()
    train_model, predict_model = qa_model.get_lstm_cnn_model(embedding_file, len(vocabulary))
    epo = 50
    if mode == 'train':
        # load training data
        qa_data = QAData()
        questions, good_answers, bad_answers = qa_data.get_training_data()

        # train the model
        Y = np.zeros(shape=(questions.shape[0],))
        train_model.fit([questions, good_answers, bad_answers], Y, epochs=epo, batch_size=64, validation_split=0.1,
                        verbose=1)

        # save the trained model
        # train_model.save_weights('model/train_weights_epoch_' + str(epo) + '.h5', overwrite=True)
        predict_model.save_weights('model/predict_weights_epoch_' + str(epo) + '.h5', overwrite=True)

    elif mode == 'predict':
        # load the evaluation data
        data = pickle.load(open("./data/dev.pkl", 'rb'))
        random.shuffle(data)

        # load weights from trained model
        qa_data = QAData()
        predict_model.load_weights('model/predict_weights_epoch_' + str(epo) + '.h5')

        c = 0
        c1 = 0
        for i, d in enumerate(data):
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
        print("Precision", precision)
        print("MRR", mrr)


if __name__ == "__main__":
    main(mode='train')
    main(mode='predict')
