import pickle

def load_pkl(path):
    file = open(path,'rb')
    return pickle.load(file)