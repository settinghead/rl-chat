import random
import string
import data
from corpus_utils import BEGIN_TAG, END_TAG
from corpus_utils import tokenize_sentence, LanguageIndex
import random

CONVO_LEN = 15
MIN_UTTERANCE_LEN = 4
MAX_UTTERANCE_LEN = 30


class Environment:

    @property
    def lang(self):
        return self._lang

    def __init__(self):
        self.reset()
        self._questions, _ = data.load_conv_text()
        self._lang = LanguageIndex(
            self._questions + [f"{BEGIN_TAG} {END_TAG}"])

    def step(self, action):
        if len(self.history) == 0:
            reward = 0
        else:
            last_from_env = self.history[-1]
            reward = self.calc_reward(action, last_from_env)

        done = len(self.history) > CONVO_LEN
        self.history.append(action)

        state = random.sample(self._questions, 1)[0]
        state = tokenize_sentence(state)[:MAX_UTTERANCE_LEN]
        state = ' '.join(state)

        self.history.append(state)

        return state, reward, done

    def reset(self):
        self.history = []

    def calc_reward(self, utterance1: str, utterance2: str):
        # calc string distance
        return SequenceMatcher(
            None, [
                self.lang.word2idx[t] for t in
                tokenize_sentence(utterance1)
            ], [
                self.lang.word2idx[t] for t in
                tokenize_sentence(utterance2)]
        ).ratio()


from difflib import SequenceMatcher


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
    while not done:
        action = "hello"
        state, reward, done = env.step(action)
        print(f"bot: {action} -> env: {state} reward: {reward}")
