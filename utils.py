import random
import string
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import spacy

BEGIN_TAG = '<GO>'
END_TAG = '<EOS>'
def load_conv_text():
    questions = []
    answers = []
    with open('conv.txt') as f:
        for line in f:
            question_answer_pair = line.split("||")
            question = question_answer_pair[0].strip()
            answer = question_answer_pair[1].strip()
            questions.append(question)
            answers.append(BEGIN_TAG + ' ' + answer + ' ' + END_TAG)
    return questions, answers


nlp = spacy.load('en')

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
    return [token.text for token in nlp(sentence)]

def get_embeddings(vocab, embedding_dim = 1024):
    url = "https://tfhub.dev/google/elmo/2"
    elmo = hub.Module(url)

    def ELMoEmbedding(elmo, x):
        return tf.reshape(elmo(x, signature="default", as_dict=True)["elmo"], [embedding_dim, ])
    vectors = []
    for lex in vocab:
        vectors.append(ELMoEmbedding(elmo, [lex]))
    return tf.stack(vectors)


