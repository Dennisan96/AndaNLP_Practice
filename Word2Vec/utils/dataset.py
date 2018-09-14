"""
This python code primily conerned with managing dataset
"""


import tensorflow as tf
import numpy as np
# import chazutsu
from nltk import tokenize
import zipfile
import tools


# the following funcitons using reference from stanford university cs20

def read_data(file_path):
    with zipfile.ZipFile(file_path) as f:
        words = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return words

def build_vocab(words, vocab_size, visual_fld):
    tools.safe_mkdir(visual_fld)
    file = open(os.path.join(visual_fld, 'vocab.tsv'), 'w')

    dictionary = dict()
    count = [('UNK', -1)]  # QUESTION: why use "UNK" and -1
    index = 0
    count.extend(Counter(words).most_common(vocab_size - 1))

    for word, _ in count:
        dictionary[word] = index
        index += 1
        file.write(word+'\n')

    index_dictionary = dict(zip(dictionary.values(), dictionary.keys())) # QUESTION: why build index dictionary
    file.close()
    return dictionary, index_dictionary

def convert_words_to_index(words, dictionary):
    """
    Here we have a list of words that represent by its index in dictionary, where the
    word in ranked by its occurance
    """

    return [dictionary[word] for word in words if word in dictionary else 0]

def generate_sample(index_words, window_size):
    """ form training paris according to skip-gram model"""
    for index, center in enumerate(index_words):
        context = random.randint(1, window_size)
        # Window size is how many word to the right/left of center word

        # get a random target before the center word
        for target in index_words[max(0, index-context): index]:
            yield center, target

        # get a random target after the center word
        for target in index_word[index+1:index+context+1]:
            yield center, target



def batch_gen(vocab_size, batch_size, skip_window, visual_fld):
    local_dest = 'data/text8.zip'
    # Here omit the downloaded part as the data will contain in the folder
    words = read_data(local_dest)
    dictionary, _ = build_vocab(words, vocab_size, visual_fld)
    index_words = convert_words_to_index(words, dictionary)
    del words
    single_gen = generate_sample(index_words, skip_window)

    while True:
        center_batch = np.zeros(batch_size, dtype=np.int32)
        target_batch = np.zeros([batch_size, 1])
        for index in range(batch_size):
            center_batch[index], target_batch[index] = next(single_gen)
        yield center_batch, target_batch


def main():
    pass

if __name__ == '__main__':
    main()
