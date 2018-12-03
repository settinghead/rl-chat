import random
import string
import data
from corpus_utils import tokenize_sentence, LanguageIndex
from itertools import takewhile
import random
from difflib import SequenceMatcher

from pycorenlp import StanfordCoreNLP
import random
import math
import pdb

CONVO_LEN = 1
MIN_UTTERANCE_LEN = 4
MAX_UTTERANCE_LEN = 20


from data import BEGIN_TAG, END_TAG, EMPTY_TOKEN, UNK_TOKEN


def char_tokenizer(s: str):
    return list(s)


class Environment:

    @property
    def lang(self):
        return self._lang

    def __init__(self):
        self.reset()
        # TO DO: this should be a new test list of questions, assuming pretrained with conv()
        self._questions, self._answers = data.load_conv_text()
        self._lang = LanguageIndex(
            self._questions + self._answers,
            #tokenizer=lambda s: list(s),
            empty_token=EMPTY_TOKEN,
            unknown_token=UNK_TOKEN
        )
        self.stanford = StanfordCoreNLP('http://localhost:9000')
        #self.stanford = StanfordCoreNLP('http://127.0.0.1:9000')
        #self.stanford = StanfordCoreNLP('http://0.0.0.0:9000')

    def step(self, action):
        reward = self.calc_reward(action)
        done = len(self.history) >= CONVO_LEN  # Present, but NOT USED
        self.history.append(action)

        # <------- TO DO: decide if randomly sampled!
        next_state = random.sample(self._questions, 1)[0]

        # state = char_tokenizer(state)[:MAX_UTTERANCE_LEN]
        next_state = ''.join(next_state)
        next_state = f'{BEGIN_TAG} {next_state} {END_TAG}'

        self.history.append(next_state)

        return next_state, reward, done

    def reset(self):
        # random.seed(48)
        self.history = []

    def calc_reward(self, utterance: str):
        # Use CoreNLP to calculate rewards
        result = self.stanford.annotate(utterance,
                                        properties={
                                            'annotators': 'sentiment',
                                            'outputFormat': 'json',
                                            'timeout': '5000'
                                        })

        # Result types from CoreNLP:
        # negative: 1; # neutral: 2; positive: 3
        s_scores = [int(s['sentimentValue']) - 2 for s in result['sentences']]
        reward = math.tanh(sum(s_scores))

        return reward



# some test code
if __name__ == "__main__":
    env = Environment()
    done = False
    action = "I love you!"
    state = ""
    # action = "hello"
    prev_state = state
    state, reward, done = env.step(action)
    print(f"env: {prev_state} -> bot: {action} reward: {reward}")
    action = state
