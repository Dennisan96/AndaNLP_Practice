import os

def safe_mkdir(dir):
    try:
        os.mkdir(dir)
    except:
        pass
