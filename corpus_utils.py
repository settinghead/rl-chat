from collections import defaultdict

BEGIN_TAG = '<GO>'
END_TAG = '<EOS>'


def tokenize_sentence(sentence):
    sentence = sentence.replace('.', ' .')
    sentence = sentence.replace(',', ' ,')
    sentence = sentence.replace('?', ' ?')
    sentence = sentence.replace('!', ' !')
    return [t for t in sentence.split(' ') if len(t) > 0]


EMPTY_IDX = 0
UNKNOWN_IDX = 1


class LanguageIndex():
    def __init__(self, samples,
                 tokenizer=tokenize_sentence,
                 empty_token='<pad>',
                 unknown_token='<UNK>'):
        self._tokenizer = tokenizer
        self.samples = samples
        self._empty_token = empty_token
        self._unknown_token = unknown_token
        self.word2idx = defaultdict(lambda: UNKNOWN_IDX)
        self.idx2word = defaultdict(lambda: unknown_token)
        self.vocab = set()
        self.create_index()

    def create_index(self):
        self.vocab.update('<UNK>')
        for phrase in self.samples:
            self.vocab.update(self._tokenizer(phrase))

        self.vocab = sorted(self.vocab)

        self.word2idx[self._empty_token] = EMPTY_IDX
        self.word2idx[self._unknown_token] = UNKNOWN_IDX
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + len([EMPTY_IDX, UNKNOWN_IDX])

        for word, index in self.word2idx.items():
            self.idx2word[index] = word
