class Model(object):
    def add_placeholders(self):
        """
        Adds placeholder.

        Placeholder are variables to represent locations where data is inserted
        """
        raise NotImplementedError("Model must implement this method")


    def create_feed_dict(self, inputs_batch, labels_batch=None):
        """
        Create feeding dict for the single step of training

        Args:
            tensor - inputs_batch - a batch of input data
            tensor - labels_batch - a batch of input data

        Returns:
            dict - feed_dict

        """
        raise NotImplementedError("Model must implement this method")


    def add_prediction_op(self):
        """
        predict using the inputs_batch

        Returns:
            tensor - pred - a tensor of shape (batch, n_classess)
        """
        raise NotImplementedError


    def add_loss_op(self, pred):
        """
        Adds Ops for the loss function to the computational graph.

        Args:
            tensor - pred - A tensor of shape (batch_size, n_classes)

        Returns:
            tensor - loss - A 0-d tensor (scalar) output
        """
        raise NotImplementedError("Each Model must re-implement this method.")


    def add_training_op(self, loss):
        """

        Sets up the training Ops.

        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        sess.run() to train the model. See


        Args:
            tensor - loss: Loss tensor (a scalar).
        Returns:
            tensor - train_op: The Op for training.
        """

        raise NotImplementedError("Each Model must re-implement this method.")

    def train_on_batch(self, session, inputs_batch, labels_batch):
        """
        Perform one gradients decent on current inputs

        Args:
            session - tf.session
            inputs_batch - np.ndarray - array of shape (batch_size, n_features)
            labels_batch - np.ndarray - array of shape (batch_size, n_features)

        Returns:
            loss - scalar - loss over curr batch

        """

        feed = self.create_feed_dict(inputs_batch, labels_batch)
        _, loss = session.run([self.train_op, self.loss], feed_dict=feed)

        return loss

    def predict_on_tatch(self, session, inputs_batch):
        """
        Predict the result given the inputs

        Args:
            session - tf.Session()
            inputs_batch - np.ndarray of shape

        Returns:
            pred

        """

        feed = self.create_feed_dict(inputs_batch)
        pred = session.run(self.pred, feed_dict=feed)
        return pred

    def build(self):
        self.add_placeholders()
        self.pred = self.add_prediction_op()
        self.loss = self.add_loss_op(self.pred)
        self.train_op = self.add_training_op(self, loss)
