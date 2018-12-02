import random
import string
import data
from corpus_utils import tokenize_sentence, LanguageIndex
from itertools import takewhile
import random
from difflib import SequenceMatcher

#from pycorenlp import StanfordCoreNLP
import random
import math
import pdb

CONVO_LEN = 1
MIN_UTTERANCE_LEN = 4
MAX_UTTERANCE_LEN = 20
BEGIN_TAG = "▶"
END_TAG = "◀"
EMPTY_TOKEN = "◌"


def char_tokenizer(s: str):
    return list(s)


class Environment:

    @property
    def lang(self):
        return self._lang

    def __init__(self):
        self.reset()
        self._questions, _ = data.load_conv_text() #TO DO: this should be a new test list of questions, assuming pretrained with conv()
        self._lang = LanguageIndex(
            [f"{BEGIN_TAG}{END_TAG}"] + self._questions,
            tokenizer=lambda s: list(s),
            empty_token=EMPTY_TOKEN
        )
        #self.stanford = StanfordCoreNLP('http://localhost:9000')

    def step(self, action):
        if len(self.history) == 0:
            reward = 0
        else:
            #last_from_env = self.history[-1]
            reward = self.calc_reward(action)
            # if reward > 0:
            #     print(last_from_env, "|", action)

        done = len(self.history) >= CONVO_LEN #Present, but NOT USED
        self.history.append(action)

        next_state = random.sample(self._questions, 1)[0] #<------- TO DO: decide if randomly sampled!
        
        # state = char_tokenizer(state)[:MAX_UTTERANCE_LEN]
        next_state = ''.join(next_state)
        next_state = f'{BEGIN_TAG}{next_state}{END_TAG}'

        self.history.append(next_state)

        return next_state, reward, done

    def reset(self):
        # random.seed(48)
        self.history = []

    def calc_reward(self, utterance: str):
        # Use CoreNLP to calculate rewards
        '''
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
        '''
        reward = 0
        
        return reward



# some test code
if __name__ == "__main__":
    env = Environment()
    done = False
    action = ""
    state = ""
    while not done:
        # action = "hello"
        prev_state = state
        state, reward, done = env.step(action)
        print(f"env: {prev_state} -> bot: {action} reward: {reward}")
        action = state
