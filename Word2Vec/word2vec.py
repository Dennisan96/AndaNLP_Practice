"""
This is a "toy" model for word2vec using Skim-gram model
"""

import tensorflow as tf
from utils.dataset import DataHandler
from models.skipgram import Skipgram
from utils.tools import visualizeWord
import matplotlib


import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2' # this is to disable a warning

SKIP_STEP = 1000

class Config():
    window_size = 1
    n_features = 128
    batch_size = 128
    vocabulary_size = 20000
    lr = 0.5
    num_sampled = 64
    epoch = 10000
    visual_fld = 'visualization'
    local_data_dest = 'data/text8.zip'


config = Config() # create configeration object to store hyperparameters

def run_word2vec(train_data):
    iterator = train_data.make_initializable_iterator()
    center_words, target_words = iterator.get_next()

    model = Skipgram(config, center_words, target_words)
    init = tf.global_variables_initializer()

    saver = tf.train.Saver() # TODO: Save the model embeddings

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

        return model.embeddings


# def gen():
#     yield from batch_gen(config.local_data_dest, config.vocabulary_size, config.batch_size, config.window_size, config.visual_fld)

def main():
    dh = DataHandler(config.local_data_dest, config.vocabulary_size, config.visual_fld, config.window_size, config.batch_size)

    dataset = tf.data.Dataset.from_generator(dh.batch_gen,
                                (tf.int32, tf.int32),
                                (tf.TensorShape([config.batch_size]), tf.TensorShape([config.batch_size, 1])))

    final_embeddings = run_word2vec(dataset)

    words_to_visualize = ['university', 'government', 'century', 'life', 'year', 'day', 'national'
                    'french', 'britsh', 'same', 'another', 'german', 'music', 'great', 'very'
                    'modern', 'common']

    visualizeWord(final_embeddings, words_to_visualize, dh)
    # print(final_embeddings.shape)

if __name__ == '__main__':
    main()
