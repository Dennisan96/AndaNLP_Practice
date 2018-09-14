import tensorflow as tf
from sklearn import preprocessing
import math

class Skipgram:

    def __init__(self, config, input_tensor, labels_tensor):
        self.config = config
        self.input = input_tensor
        self.labels = labels_tensor
        self.build()


    def add_placeholders(self):
        # Input is a index of center words
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size))

        # Label is a index of target_words
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, 1))

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {self.inputs_placeholder: inputs_batch, self.labels_placeholder: labels_batch}
        return feed_dict

    def add_training_op(self):
        embedding = tf.Variable(tf.random_uniform([self.config.vocabulary_size, self.config.n_features], -1, 1))

        batch_embedding = tf.nn.embedding_lookup(embedding, self.input)

        nceweights = tf.Variable(tf.truncated_normal([self.config.vocabulary_size, self.config.n_features], stddev=1.0/math.sqrt(self.config.n_features)))

        ncebias = tf.Variable(tf.zeros([self.config.vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=nceweights,
                                            biases=ncebias,
                                            labels=self.labels,
                                            inputs=batch_embedding,
                                            num_sampled=self.config.num_sampled,
                                            num_classes=self.config.vocabulary_size))

        train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)

        return loss, train_op

    def build(self):
        self.loss, self.train_op = self.add_training_op()
