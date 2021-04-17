import os
from os.path import join as pjoin
import json
import datetime
import tensorflow as tf
import logging
from models.BiDAF import BiDAF
from models.Baseline import Baseline
from models.Attention import LuongAttention
from utils.data_reader import load_and_preprocess_data, load_word_embeddings
from utils.result_saver import ResultSaver

logging.basicConfig(level=logging.INFO)

tf.app.flags.DEFINE_float("learning_rate", 0.0048, "Learning rate")
tf.app.flags.DEFINE_float("keep_prob", 0.75, "The probably that a node is kept after the affine transform")
tf.app.flags.DEFINE_float("max_grad_norm", 5.,
                          "The maximum grad norm during backpropagation, anything greater than max_grad_norm is truncated to be max_grad_norm")
tf.app.flags.DEFINE_integer("batch_size", 24, "Number of batches to be used per training batch")
tf.app.flags.DEFINE_integer("eval_num", 250, "Evaluate on validation set for every eval_num batches trained")
tf.app.flags.DEFINE_integer("embedding_size", 100, "Word embedding size")
tf.app.flags.DEFINE_integer("window_size", 3, "Window size for sampling during training")
tf.app.flags.DEFINE_integer("hidden_size", 100, "Hidden size of the RNNs")
tf.app.flags.DEFINE_integer("samples_used_for_evaluation", 500,
                            "Samples to be used at evaluation for every eval_num batches trained")
tf.app.flags.DEFINE_integer("num_epochs", 10, "Number of Epochs")
tf.app.flags.DEFINE_integer("max_context_length", None, "Maximum length for the context")
tf.app.flags.DEFINE_integer("max_question_length", None, "Maximum length for the question")
tf.app.flags.DEFINE_string("data_dir", "data/squad", "Data directory")
tf.app.flags.DEFINE_string("train_dir", "train/{}".format(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
), "Saved training parameters directory")
tf.app.flags.DEFINE_string("retrain_embeddings", False, "Whether or not to retrain the embeddings")
tf.app.flags.DEFINE_string("share_encoder_weights", False, "Whether or not to share the encoder weights")
tf.app.flags.DEFINE_string("learning_rate_annealing", False, "Whether or not to anneal the learning rate")
tf.app.flags.DEFINE_string("ema_for_weights", False, "Whether or not to use EMA for weights")
tf.app.flags.DEFINE_string("log", True, "Whether or not to log the metrics during training")
tf.app.flags.DEFINE_string("optimizer", "adam", "The optimizer to be used ")
tf.app.flags.DEFINE_string("model", "BiDAF", "Model type")
tf.app.flags.DEFINE_string("find_best_span", True, "Whether find the span with the highest probability")

FLAGS = tf.app.flags.FLAGS


def initialize_model(session, train_dir):
    if not os.path.exists(train_dir):
        session.run(tf.global_variables_initializer())
        os.makedirs(train_dir, exist_ok=True)

        # Save the config file
        with open(pjoin(FLAGS.train_dir, "config.txt"), "w") as f:
            output = ""
            for k, v in FLAGS.__flags.items():
                output += "{} : {}\n".format(k, v)
            f.write(output)
    else:
        saver = tf.train.Saver()
        checkpoint = tf.train.get_checkpoint_state(train_dir)
        saver.restore(session, checkpoint.model_checkpoint_path)

def main(_):
    # load the data
    train, val = load_and_preprocess_data(FLAGS.data_dir)

    # load the word matrix
    embeddings = load_word_embeddings(FLAGS.data_dir)

    # Create the saver helper object
    result_saver = ResultSaver(FLAGS.train_dir)

    if FLAGS.model == "BiDAF":
        model = BiDAF(result_saver, embeddings, FLAGS)
    elif FLAGS.model == "Baseline":
        model = Baseline(result_saver, embeddings, FLAGS)
    elif FLAGS.model == "LuongAttention":
        model = LuongAttention(result_saver, embeddings, FLAGS)

    logging.info("Start training with hyper parameters:")

    print(vars(FLAGS)["__flags"])

    with tf.Session() as sess:
        initialize_model(sess, FLAGS.train_dir)
        model.train(sess, train, val)


if __name__ == "__main__":
    tf.app.run()
