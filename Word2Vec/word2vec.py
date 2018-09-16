"""
This is a "toy" model for word2vec using Skim-gram model
"""

import tensorflow as tf
from utils.dataset import DataHandler
from models.skipgram import Skipgram
from utils.tools import visualizeWord
import matplotlib
import numpy as np
from tensorflow.python.tools import inspect_checkpoint as chkp


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # this is to disable a warning

SKIP_STEP = 1000

class Config():
    window_size = 1
    n_features = 150
    batch_size = 128
    vocabulary_size = 50000
    lr = 0.8
    num_sampled = 120
    epoch = 100000
    visual_fld = 'visualization'
    local_data_dest = 'data/text8.zip'


config = Config() # create configeration object to store hyperparameters


def run_word2vec(train_data):
    iterator = train_data.make_initializable_iterator()
    center_words, target_words = iterator.get_next()

    model = Skipgram(config, center_words, target_words)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)
        sess.run(iterator.initializer)
        average_loss = 0.0

        for i in range(config.epoch):
            try:
                loss_batch, _ = sess.run([model.loss, model.train_op])
                average_loss += loss_batch

                if (i+1)%SKIP_STEP == 0:
                    average_loss /= SKIP_STEP
                    print('Average loss at step {}:{:5.1f}'.format(i+1, average_loss))
                    average_loss = 0;
            except tf.errors.OutOfRangeError:
                sess.run(iterator.initializer)

        save_path = saver.save(sess, './saved/model.ckpt')
        print("The model is saved under path:", save_path)


def main():
    # create a Data Handle to manage the training data
    dh = DataHandler(config.local_data_dest, config.vocabulary_size, config.visual_fld, config.window_size, config.batch_size)

    dataset = tf.data.Dataset.from_generator(dh.batch_gen,
                                (tf.int32, tf.int32),
                                (tf.TensorShape([config.batch_size]), tf.TensorShape([config.batch_size, 1])))

    # start training the model
    run_word2vec(dataset)

    # restore saved model to get final words vectors
    tf.reset_default_graph()
    embed = tf.get_variable("embeddings", shape=[config.vocabulary_size, config.n_features])
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, './saved/model.ckpt')
        final_embeddings = embed.eval()


    # pick some words to visulize on a graph
    words_to_visualize = ['university', 'government', 'century', 'life', 'year', 'day', 'national'
                    'french', 'british', 'same', 'another', 'music', 'great', 'very'
                    'modern', 'common']

    visualizeWord(final_embeddings, words_to_visualize, dh)


if __name__ == '__main__':
    main()
