import tensorflow as tf
from sklearn import preprocessing
import math

class Skipgram:

    def __init__(self, config, input_tensor, labels_tensor):
        self.config = config
        self.input = input_tensor
        self.labels = labels_tensor
        self.build()

    def add_training_op(self):
        embeddings = tf.Variable(tf.random_uniform([self.config.vocabulary_size, self.config.n_features], -1, 1))

        batch_embedding = tf.nn.embedding_lookup(embeddings, self.input)

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
