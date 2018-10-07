class Environment:
    def __init__(self):
        self.reset()

    def step(self, action):
        reward = calc_reward(action, self.history)
        state = self.history
        done = len(self.history) > 10
        self.history.push(action)

        return state, reward, done

    def reset(self):
        self.history = []


def calc_reward(action: str, history: str[]):
    # last sentence (from environment/user)
    last_env = history[-1]
