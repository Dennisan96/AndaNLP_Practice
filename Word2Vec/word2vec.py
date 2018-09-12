"""
This is a "toy" model for word2vec using Skim-gram model
"""

import tensorflow as tf
from util.dataset import getSampleText
from util.dataset import paragraphToSentence


class Config():
    window_size = 1
    n_features = 100
    batch_size = 50
    lr = 0.005

class skimGram(Model):

    def add_placeholders(self):
