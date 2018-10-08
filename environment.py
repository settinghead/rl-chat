CONVO_LEN = 15


class Environment:
    def __init__(self):
        self.reset()

    def step(self, action):
        last_from_env = self.history[-1]
        reward = calc_reward(action, last_from_env)

        done = len(self.history) > CONVO_LEN
        self.history.append(action)

        state = random_utterance()
        self.history.append(action)

        return state, reward, done

    def reset(self):
        self.history = ["hello"]


from difflib import SequenceMatcher


def calc_reward(utterance1: str, utterance2: str):
    # calc string distance
    return SequenceMatcher(
        None, utterance1, utterance2
    ).ratio()


N = 20

import random
import string


def random_utterance():
    return ''.join(random.choice(string.ascii_uppercase + string.digits)
                   for _ in range(N))


# some test code
if __name__ == "__main__":
    env = Environment()
    done = False
    while not done:
        state, reward, done = env.step("hello")
        print(state, reward)
