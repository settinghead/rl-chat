from collections import defaultdict
import nltk
import itertools

# space is included in whitelist
'''
BEGIN_TAG = "▶"
END_TAG = "◀"
EMPTY_TOKEN = "◌"
'''
BEGIN_TAG = "<go>"
END_TAG = "<eos>"
EMPTY_TOKEN = "◌"


EN_WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyz .<>.,?:;!&[]' + \
    BEGIN_TAG + END_TAG + EMPTY_TOKEN
EN_BLACKLIST = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~\''


def tokenize_sentence(sentence):
    sentence = sentence.lower()
    sentence = sentence.replace('.', ' .')
    sentence = sentence.replace(',', ' ,')
    sentence = sentence.replace('?', ' ?')
    sentence = sentence.replace('!', ' !')
    return [t for t in sentence.split(' ') if len(t) > 0]


EMPTY_IDX = 0
UNKNOWN_IDX = 1
VOCAB_SIZE = 10000
limit = {
    'maxq': 30,
    'minq': 0,
    'maxa': 30,
    'mina': 3
}


def filter_line(line):
    return ''.join([ch for ch in line if ch in EN_WHITELIST])


def index_(tokenized_sentences, vocab_size, extra_vocab):
    # get frequency distribution
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    # get vocabulary of 'vocab_size' most used words
    vocab = freq_dist.most_common(vocab_size)
    # index2word
    index2word = extra_vocab + [x[0] for x in vocab]
    # word2index
    word2index = dict([(w, i) for i, w in enumerate(index2word)])
    return index2word, word2index


class LanguageIndex():
    def __init__(self, samples,
                 tokenizer=tokenize_sentence,
                 empty_token='<pad>',
                 unknown_token='<unk>'):
        self._tokenizer = tokenizer
        self.empty_token = empty_token
        self._unknown_token = unknown_token
        self.idx2word, self.word2idx = index_(
            [
                tokenizer(s) for s in samples
            ], VOCAB_SIZE, [unknown_token, empty_token]
        )
        self.vocab = self.idx2word
