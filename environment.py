CONVO_LEN = 15


class Environment:
    def __init__(self):
        self.reset()

    def step(self, action):
        if len(self.history) == 0:
            reward = 0
        else:
            last_from_env = self.history[-1]
            reward = calc_reward(action, last_from_env)

        done = len(self.history) > CONVO_LEN
        self.history.append(action)

        state = random_utterance()
        self.history.append(state)

        return state, reward, done

    def reset(self):
        self.history = []


from difflib import SequenceMatcher


def calc_reward(utterance1: str, utterance2: str):
    # calc string distance
    return SequenceMatcher(
        None, utterance1, utterance2
    ).ratio()


MIN_UTTERANCE_LEN = 4
MAX_UTTERANCE_LEN = 20

import random
import string


def random_utterance():
    utt_len = random.randint(MIN_UTTERANCE_LEN, MAX_UTTERANCE_LEN + 1)
    random_chars = [random.choice(
        string.ascii_uppercase +
        string.digits +
        string.ascii_lowercase + ' .,!?'
    ) for _ in range(utt_len)]
    result = ''.join(random_chars)
    return result


# some test code
if __name__ == "__main__":
    env = Environment()
    done = False
    while not done:
        action = "hello"
        state, reward, done = env.step(action)
        print(f"bot: {action} -> env: {state} reward: {reward}")
