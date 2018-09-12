"""
This is a "toy" model for word2vec using Skim-gram model
"""

import tensorflow as tf
from utils.dataset import load_data
from models.skimgram import Skimgram


class Config():
    window_size = 1
    n_features = 128
    batch_size = 128
    vocabulary_size = 10000
    lr = 0.005

def run(train_data):
    config = Config()
    model = Skimgram(config)


    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)

        average_loss = 0.0
        for step, batch_data in enumerate(train_data):
            feed_dict = model.create_feed_dict(batch_data)
            _, loss_val = sess.run([model.train_op, model.loss], feed_dict)
            average_loss += loss_val

            if step%1000 == 0:
                if step > 0:
                    average_loss /= 1000
                print("loss at iter", step, ":", average_loss)




def main():
    train_data = load_data()
    run(train_data)


if __name__ == '__main__':
    main()
