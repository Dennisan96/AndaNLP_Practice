"""
This is a "toy" model for word2vec using Skim-gram model
"""

import tensorflow as tf
from utils.dataset import batch_gen
from models.skipgram import Skipgram

SKIP_STEP = 1000

class Config():
    window_size = 1
    n_features = 128
    batch_size = 128
    vocabulary_size = 50000
    lr = 0.5
    num_sampled = 64
    epoch = 100000
    visual_fld = 'visualization'
    local_data_dest = 'data/text8.zip'

config = Config() # create configeration object to store hyperparameters

def run_word2vec(train_data):
    iterator = train_data.make_initializable_iterator()
    center_words, target_words = iterator.get_next()

    model = Skipgram(config, center_words, target_words)
    init = tf.global_variables_initializer()
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



def gen():
    yield from batch_gen(config.local_data_dest, config.vocabulary_size, config.batch_size, config.window_size, config.visual_fld)


def main():
    config = Config()
    dataset = tf.data.Dataset.from_generator(gen,
                                (tf.int32, tf.int32),
                                (tf.TensorShape([config.batch_size]), tf.TensorShape([config.batch_size, 1])))
    run_word2vec(dataset)


if __name__ == '__main__':
    main()
