import random
import string
import data
from corpus_utils import tokenize_sentence, LanguageIndex
from itertools import takewhile
import random
from difflib import SequenceMatcher
import transformer.Constants as Constants

CONVO_LEN = 1
MIN_UTTERANCE_LEN = 4
MAX_UTTERANCE_LEN = 20
BEGIN_TAG = "▶"
END_TAG = "◀"
EMPTY_TOKEN = "◌"
UNKNOWN_TOKEN = "♡"

def char_tokenizer(s: str):
    return list(s)


class Environment:

    @property
    def lang(self):
        return self._lang

    def __init__(self):
        self.reset()
        self._questions, _ = data.load_conv_text()
        self._lang = LanguageIndex(
            self._questions
        )

    def step(self, action):
        if len(self.history) == 0:
            reward = 0
        else:
            last_from_env = self.history[-1]
            reward = self.calc_reward(action, last_from_env)
            # if reward > 0:
            #     print(last_from_env, "|", action)

        done = len(self.history) >= CONVO_LEN
        self.history.append(action)

        state = random.sample(self._questions, 1)[0]
        # state = "hellozzz"
        #state = char_tokenizer(state)[:MAX_UTTERANCE_LEN]
        #state = ''.join(state)
        #state = f'{BEGIN_TAG}{state}{END_TAG}'

        self.history.append(state)

        return state, reward, done

    def reset(self):
        # random.seed(48)
        self.history = []

    def calc_reward(self,
        utterance1: str,
        utterance2: str,
        exclude_tokens = [Constants.EOS, Constants.PAD, Constants.BOS]):
        # calc string distance
        token_seq1 = [self.lang.word2idx[t] for t in tokenize_sentence(utterance1)]
        token_seq2 = [self.lang.word2idx[t] for t in tokenize_sentence(utterance2)]
        seq1 = [t for t in token_seq1 if t not in exclude_tokens] 
        seq2 = [t for t in token_seq2 if t not in exclude_tokens]
        r = SequenceMatcher(None, seq1, seq2).ratio()
        # if(r > 0):
        #     print([self.lang.idx2word[idx]
        #            for idx in set(seq2).intersection(set(seq1))])
        return r

# def random_utterance(min_len, max_len):
#     utt_len = random.randint(min_len, max_len + 1)
#     random_chars = [random.choice(
#         string.ascii_uppercase +
#         string.digits +
#         string.ascii_lowercase + ' .,!?'
#     ) for _ in range(utt_len)]
#     result = ' '.join(random_chars)
#     result = BEGIN_TAG + ' ' + result + ' ' + END_TAG
#     return result


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
