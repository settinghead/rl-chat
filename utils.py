import random
import string


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
            

def random_utterance(min_len, max_len):
    utt_len = random.randint(min_len, max_len + 1)
    random_chars = [random.choice(
        string.ascii_uppercase +
        string.digits +
        string.ascii_lowercase + ' .,!?'
    ) for _ in range(utt_len)]
    result = ' '.join(random_chars)
    result = BEGIN_TAG + ' ' + result + ' ' + END_TAG
    print(result)
    return result

import spacy
import numpy as np

def get_embeddings(vocab):
    nlp = spacy.load('en')
    nlp.vocab.vectors.from_glove('/Users/renyuli/MLProject/rl-chat/glove.6B/')
    ''' the function to retreive the embedding space from Spacy GloVe
    Vocab is the vocab built from your text'''
    nr_vector = len(vocab) + 1 # 1 is saved for padding!!
    # preset the embedding matrix, 300 is the embedding vector length
    vectors = np.zeros((nr_vector, 300), dtype='float32')
    for i, lex in enumerate(vocab):
        if lex in nlp.vocab:
            print(nlp.vocab[lex].vector)
            # the 1st word is saved for padding!!!!
            vectors[i + 1] = nlp.vocab[lex].vector / nlp.vocab[lex].vector_norm

    return vectors
