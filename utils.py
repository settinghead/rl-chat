import random
import string
import tensorflow as tf
import numpy as np

# def get_ELMo_embeddings():
#     url = "https://tfhub.dev/google/elmo/2"
#     elmo = hub.Module(url)
#     return elmo


def max_length(tensor):
    return max(len(t) for t in tensor)


def get_GloVe_embeddings(vocab, embedding_dim):
    embeddings = dict()
    f = open('pretrained_models/glove.6B.'+str(embedding_dim)+'d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs

    embedding_matrix = np.zeros((len(vocab) + 1, embedding_dim))
    for i, word in enumerate(vocab):
        if word in embeddings.keys():
            embedding_matrix[i+1] = embeddings[word]
    return embedding_matrix
