import tensorflow as tf
from tensorflow.keras.layers import layers
from keras import backend as K
from tensorflow.keras.layers import Embedding, LSTM, Input, Lambda, concatenate, Dot, Conv1D, Bidirectional
# from tensorflow.keras.layers.convolutional import Convolution1D
from tensorflow.keras.models import Model
import numpy as np
# from tensorflow.keras.layers.wrappers import Bidirectional
import pandas as pd

from tensorflow.keras.layers import GlobalAveragePooling1D, GlobalMaxPooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
import transformers

class QAModel:
    @staticmethod
    def get_lstm_cnn_model(embedding_file):
        margin = 0.2
        hidden_dim = 141
        sentence_length = 200
        weights = np.load(embedding_file)
        weights = pd.DataFrame(weights)
        weights = weights.loc[(weights != 0).any(axis=1)]
        weights = weights.to_numpy()
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
        question_pool = concatenate([qf_rnn, qb_rnn], axis=-1)
        af_rnn = f_rnn(answer_embedding)
        ab_rnn = b_rnn(answer_embedding)
        answer_pool = concatenate([af_rnn, ab_rnn], axis=-1)

        filter_sizes = [1, 2, 3, 5]
        cnns = [Conv1D(filters=500, kernel_size=ngram_size, activation='tanh', padding='same')
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

    @staticmethod
    def get_bert_model(max_length, ):
        # Encoded token ids from BERT tokenizer.
        input_ids = Input(
            shape=(max_length,), dtype=tf.int32, name="input_ids"
        )
        # Attention masks indicates to the model which tokens should be attended to.
        attention_masks = Input(
            shape=(max_length,), dtype=tf.int32, name="attention_masks"
        )
        # Token type ids are binary masks identifying different sequences in the model.
        token_type_ids = Input(
            shape=(max_length,), dtype=tf.int32, name="token_type_ids"
        )
        # Loading pretrained BERT model.
        bert_model = transformers.TFBertModel.from_pretrained("bert-base-uncased")
        # Freeze the BERT model to reuse the pretrained features without modifying them.
        bert_model.trainable = False

        sequence_output, pooled_output = bert_model(
            input_ids, attention_mask=attention_masks, token_type_ids=token_type_ids
        )
        # Add trainable layers on top of frozen layers to adapt the pretrained features on the new data.
        bi_lstm = Bidirectional(
            LSTM(64, return_sequences=True)
        )(sequence_output)
        # Applying hybrid pooling approach to bi_lstm sequence output.
        avg_pool = GlobalAveragePooling1D()(bi_lstm)
        max_pool = GlobalMaxPooling1D()(bi_lstm)
        concat = concatenate([avg_pool, max_pool])
        dropout = Dropout(0.3)(concat)
        output = Dense(3, activation="softmax")(dropout)
        model = Model(
            inputs=[input_ids, attention_masks, token_type_ids], outputs=output
        )

        model.compile(
            optimizer = Adam(),
            loss="categorical_crossentropy",
            metrics=["acc"],
        )
        return model


    @staticmethod
    def train_bert_model(embedding_file, batch_size):
        margin = 0.2
        # hidden_dim = 141
        max_length = 200

        # Preprocessing
        weights = np.load(embedding_file)
        weights = pd.DataFrame(weights)
        # Drop words that don't exist in the pretrained word embedding vocabulary (rows of all 0's)
        weights = weights.loc[(weights != 0).any(axis=1)]


        class BertSemanticDataGenerator(tf.keras.utils.Sequence):
            """Generates batches of data.
            Args:
                sentence_pairs: Array of premise and hypothesis input sentences.
                labels: Array of labels.
                batch_size: Integer batch size.
                shuffle: boolean, whether to shuffle the data.
                include_targets: boolean, whether to incude the labels.
            Returns:
                Tuples `([input_ids, attention_mask, `token_type_ids], labels)`
                (or just `[input_ids, attention_mask, `token_type_ids]`
                if `include_targets=False`)
            """

            def __init__(
                self,
                sentence_pairs,
                labels,
                batch_size=batch_size,
                shuffle=True,
                include_targets=True,
            ):
                self.sentence_pairs = sentence_pairs
                self.labels = labels
                self.shuffle = shuffle
                self.batch_size = batch_size
                self.include_targets = include_targets
                # Load our BERT Tokenizer to encode the text.
                # We will use base-base-uncased pretrained model.
                self.tokenizer = transformers.BertTokenizer.from_pretrained(
                    "bert-base-uncased", do_lower_case=True
                )
                self.indexes = np.arange(len(self.sentence_pairs))
                self.on_epoch_end()

            def __len__(self):
                # Denotes the number of batches per epoch.
                return len(self.sentence_pairs) // self.batch_size

            def __getitem__(self, idx):
                # Retrieves the batch of index.
                indexes = self.indexes[idx * self.batch_size : (idx + 1) * self.batch_size]
                sentence_pairs = self.sentence_pairs[indexes]

                # With BERT tokenizer's batch_encode_plus batch of both the sentences are
                # encoded together and separated by [SEP] token.
                encoded = self.tokenizer.batch_encode_plus(
                    sentence_pairs.tolist(),
                    add_special_tokens=True,
                    max_length=max_length,
                    return_attention_mask=True,
                    return_token_type_ids=True,
                    pad_to_max_length=True,
                    return_tensors="tf",
                )

                # Convert batch of encoded features to numpy array.
                input_ids = np.array(encoded["input_ids"], dtype="int32")
                attention_masks = np.array(encoded["attention_mask"], dtype="int32")
                token_type_ids = np.array(encoded["token_type_ids"], dtype="int32")

                # Set to true if data generator is used for training/validation.
                if self.include_targets:
                    labels = np.array(self.labels[indexes], dtype="int32")
                    return [input_ids, attention_masks, token_type_ids], labels
                else:
                    return [input_ids, attention_masks, token_type_ids]

            def on_epoch_end(self):
                # Shuffle indexes after each epoch if shuffle is set to True.
                if self.shuffle:
                    np.random.RandomState(42).shuffle(self.indexes)




        # training
        
        callbacks = [EarlyStopping(monitor='val_loss', patience=20),
                     ModelCheckpoint(filepath='best_model.h5', monitor='val_loss', save_best_only=True)]
        # train the model
        Y = np.zeros(shape=(questions.shape[0],))
        train_model.fit([questions, good_answers, bad_answers], Y, epochs=epo, batch_size=64, validation_split=0.1,
                        verbose=1, callbacks=callbacks)


        """
        The value "-" appears as part of our training and validation targets.
        We will skip these samples.
        """
        train_df = (
            train_df[train_df.similarity != "-"]
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )
        valid_df = (
            valid_df[valid_df.similarity != "-"]
            .sample(frac=1.0, random_state=42)
            .reset_index(drop=True)
        )


        train_data = BertSemanticDataGenerator(
            train_df[["sentence1", "sentence2"]].values.astype("str"),
            y_train,
            batch_size=batch_size,
            shuffle=True,
        )
        valid_data = BertSemanticDataGenerator(
            valid_df[["sentence1", "sentence2"]].values.astype("str"),
            y_val,
            batch_size=batch_size,
            shuffle=False,
        )
