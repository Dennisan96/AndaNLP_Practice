"""
This python code primily conerned with managing dataset
"""


import tensorflow as tf
import numpy as np
import zipfile
import utils.tools
import os
from collections import Counter
import random


# the following funcitons using reference from stanford university cs20
class DataHandler():

    def read_data(self, file_path):
        with zipfile.ZipFile(file_path) as f:
            words = tf.compat.as_str(f.read(f.namelist()[0])).split()
        return words

    def build_vocab(self, words, vocab_size, visual_fld):
        utils.tools.safe_mkdir(visual_fld)
        file = open(os.path.join(visual_fld, 'vocab.tsv'), 'w')

        dictionary = dict()
        count = [('UNK', -1)]
        index = 0
        count.extend(Counter(words).most_common(vocab_size - 1+100))
        count = count[100:] # the most frequent word are mostly connecting words

        for word, _ in count:
            dictionary[word] = index
            index += 1
            file.write(word+'\n')

        index_dictionary = dict(zip(dictionary.values(), dictionary.keys())) # QUESTION: why build index dictionary
        file.close()
        return dictionary, index_dictionary

    def convert_words_to_index(self, words):
        """
        Here we have a list of words that represent by its index in dictionary, where the
        word in ranked by its occurance
        """

        return [self.dictionary[word] if word in self.dictionary else 0 for word in words]

    def generate_sample(self, index_words, window_size):
        """ form training paris according to skip-gram model"""
        for index, center in enumerate(index_words):
            context = random.randint(1, window_size)
            # Window size is how many word to the right/left of center word

            # get a random target before the center word
            for target in index_words[max(0, index-context): index]:
                yield center, target

            # get a random target after the center word
            for target in index_words[index+1:index+context+1]:
                yield center, target


    def __init__(self, file_path, vocab_size, visual_fld, skip_window, batch_size):
        self.vocab_size = vocab_size
        self.visual_fld = visual_fld
        self.skip_window = skip_window
        self.batch_size = batch_size
        self.words = self.read_data(file_path)
        self.dictionary, self.index_dictionary = self.build_vocab(self.words, self.vocab_size, self.visual_fld)
        self.index_words = self.convert_words_to_index(self.words)




    def batch_gen(self):
        single_gen = self.generate_sample(self.index_words, self.skip_window)
        while True:
            center_batch = np.zeros(self.batch_size, dtype=np.int32)
            target_batch = np.zeros([self.batch_size, 1])
            for index in range(self.batch_size):
                center_batch[index], target_batch[index] = next(single_gen)
            yield center_batch, target_batch

def main():
    pass
if __name__ == '__main__':
    main()
