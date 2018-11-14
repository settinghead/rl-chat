from collections import defaultdict


def tokenize_sentence(sentence):
    sentence = sentence.lower()
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
                 unknown_token='<unk>'):
        self._tokenizer = tokenizer
        self.samples = samples
        self.empty_token = empty_token
        self._unknown_token = unknown_token
        self.word2idx = defaultdict(lambda: UNKNOWN_IDX)
        self.idx2word = {}
        self.create_index()

    def create_index(self):
        vocab = set()
        for phrase in self.samples:
            phrase = phrase.lower()
            vocab.update(self._tokenizer(phrase))

        sorted_vocab = sorted(vocab)

        self.word2idx[self.empty_token] = EMPTY_IDX
        self.word2idx[self._unknown_token] = UNKNOWN_IDX
        prefix = [EMPTY_IDX, UNKNOWN_IDX]
        for index, word in enumerate(sorted_vocab):
            self.word2idx[word] = len(prefix) + index

        for word, index in self.word2idx.items():
            self.idx2word[index] = word

        self.vocab = prefix + sorted_vocab

# class LanguageIndex():
#     def __init__(self, samples, tokenizer=tokenize_sentence, empty_token='<pad>'):
#         self._tokenizer = tokenizer
#         self.samples = samples
#         self.word2idx = {}
#         self.idx2word = {}
#         self.vocab = set()
#         self._empty_token = empty_token
#         self.create_index()

#     def create_index(self):
#         for phrase in self.samples:
#             self.vocab.update(self._tokenizer(phrase))

#         self.vocab = sorted(self.vocab)

#         self.word2idx[self._empty_token] = 0
#         for index, word in enumerate(self.vocab):
#             self.word2idx[word] = index + 1

#         for word, index in self.word2idx.items():
#             self.idx2word[index] = word
