import tensorflow as tf
from sklearn import preprocessing

class Skipgram:

    def __init__(self, config):
        self.config = config
        self.build()


    def add_placeholders(self):
        self.inputs_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.config.n_features))
        self.labels_placeholder = tf.placeholder(tf.int32, shape=(self.config.batch_size, self.config.n_features))

    def create_feed_dict(self, inputs_batch, labels_batch=None):
        feed_dict = {self.input_placeholder: input_batch, self.labels_placeholder: labels_batch}
        return feed_dict

    def add_training_op(self):
        embedding = tf.Variable(tf.random_uniform([self.config.vocabulary_size, self.config.n_features], -1, 1))

        batch_embedding = tf.nn.embedding_lookup(embedding, self.input_placeholder)

        weights = tf.Variable(tf.truncated_normal([self.config.vocabulary_size, self.config.n_features], stddev=1.0/math.sqrt(self.config.n_features)))

        bias = tf.Variable(tf.zeros([self.config.vocabulary_size]))

        loss = tf.reduce_mean(tf.nn.nce_loss(weights=weights,
                                            biases=bias,
                                            labels=self.labels_placeholder,
                                            inputs=self.inputs_placeholder,
                                            num_sampled=self.config.num_sampled,
                                            num_classes=self.config.vocabulary_size))

        train_op = tf.train.GradientDescentOptimizer(self.config.lr).minimize(loss)

        normalized_embedding = preprocessing.normalize(embedding)

        return normalized_embedding, loss, train_op

    def build(self):
        self.add_placeholders()
        self.embedding, self.loss, self.train_op = self.add_training_op()
