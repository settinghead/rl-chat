
BEGIN_TAG = '<GO>'
END_TAG = '<EOS>'


def tokenize_sentence(sentence):
    sentence = sentence.replace('.', ' .')
    sentence = sentence.replace(',', ' ,')
    sentence = sentence.replace('?', ' ?')
    sentence = sentence.replace('!', ' !')
    return [t for t in sentence.split(' ') if len(t) > 0]


class LanguageIndex():
    def __init__(self, samples, tokenizer=tokenize_sentence, empty_token='<pad>'):
        self._tokenizer = tokenizer
        self.samples = samples
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self._empty_token = empty_token
        self.create_index()

    def create_index(self):
        for phrase in self.samples:
            self.vocab.update(self._tokenizer(phrase))

        self.vocab = sorted(self.vocab)

        self.word2idx[self._empty_token] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word
