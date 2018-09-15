import os
import matplotlib.pyplot as plt
import numpy as np

def safe_mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass

def visualizeWord(embedding, words, dh):
    wordsIndex = dh.convert_words_to_index(words)
    visualizeVecs = [embedding[i] for i in wordsIndex]

    temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
    covariance = 1.0 / len(wordsIndex) * temp.T.dot(temp)
    U,S,V = np.linalg.svd(covariance)
    coord = temp.dot(U[:,0:2])

    for i in range(len(words)):
        plt.text(coord[i,0], coord[i,1], words[i],
            bbox=dict(facecolor='green', alpha=0.05))

    plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
    plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

    plt.savefig('q3_word_vectors.png')
