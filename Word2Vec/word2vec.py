"""
This is a "toy" model for word2vec using Skim-gram model
"""

import tensorflow as tf
from utils.dataset import batch_gen
from models.skimgram import Skimgram


class Config():
    window_size = 1
    n_features = 128
    batch_size = 128
    vocabulary_size = 10000
    lr = 0.005
    num_sampled = 64
    epoach = 10000
    visual_fld = 'visualization'

config = Config() # create configeration object to store hyperparameters

def run_word2vec(train_data):
    iterator = train_data.make_initializable_iterator()
    center_words, target_words = iterator.get_next()

    model = Skimgram(config)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        sess.run(iterator.initializer)

        average_loss = 0.0



def gen():
    yield from batch_gen(config.vocabulary_size, config.batch_size, config.window_size, config.visual_fld)


def main():
    config = Config()
    train_dataset = tf.data.Dataset.from_generator(gen, (tf.int32, tf.int32), tf.TensorShape([config.batch_size]), tf.TensorShape([config.batch_size, 1]))
    run_word2vec(train_dataset)


if __name__ == '__main__':
    main()
