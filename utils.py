import random
import string
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np

BEGIN_TAG = '<GO>'
END_TAG = '<EOS>'
def load_conv_text():
    questions = []
    answers = []
    with open('conv1.txt') as f:
        for line in f:
            question_answer_pair = line.split("||")
            question = question_answer_pair[0].strip()
            answer = question_answer_pair[1].strip()
            questions.append(question)
            answers.append(BEGIN_TAG + ' ' + answer + ' ' + END_TAG)
    return questions, answers

class LanguageIndex():
    def __init__(self, samples):
        self.samples = samples
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.create_index()

    def create_index(self):
        for phrase in self.samples:
            self.vocab.update(tokenize_sentence(phrase))

        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

def max_length(tensor):
    return max(len(t) for t in tensor)


def tokenize_sentence(sentence):
    sentence = sentence.replace('.', ' .')
    sentence = sentence.replace(',', ' ,')
    sentence = sentence.replace('?', ' ?')
    sentence = sentence.replace('!', ' !')
    return [t for t in sentence.split(' ')]

def get_ELMo_embeddings():
    url = "https://tfhub.dev/google/elmo/2"
    elmo = hub.Module(url)
    return elmo

def get_GloVe_embeddings(vocab, embedding_dim):
    embeddings = dict()
    f = open('glove.6B/glove.6B.'+str(embedding_dim)+'d.txt')
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

