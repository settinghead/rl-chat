import numpy as np


def get_GloVe_embeddings(vocab: list, embedding_dim):
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


def get_embedding_dim(use_glove: bool):
    if use_glove:
        # 200 if using glove
        return 100
    else:
        # 256 if without pretrained embedding
        return 100
