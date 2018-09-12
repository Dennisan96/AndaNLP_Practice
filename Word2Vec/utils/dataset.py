"""
This python code primily conerned with managing dataset
"""

import chazutsu
from nltk import tokenize

def getSampleText():
    """
    "Toy" vision of get sample data to prove the model correctness

    Return:
        str - sample text data
    """
    
    return ""

def paragraphToSentence(p):
    return tokenize.sent_tokenize(p)
