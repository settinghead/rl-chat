import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from encoder_decoder import Encoder, Decoder
from environment import Environment

tf.enable_eager_execution()

EPISODES = 100


class Agent:
    def __init__(self):
        pass

    def encoder_z(self, state):
        pass


def main():
    env = Environment()
    agent = Agent()

    for episode in range(EPISODES):

        state = env.reset()

        acc_rewards = []
        acc_actions = []

        while True:
            encoder_z = agent.encoder_z(state)
            actions_with_probs = beam_search(encoder_z)
            actions_dist = Categorical(actions_with_probs)

            action = actions_dist.sample()

            loss = get_loss(actions_dist)

            state, reward, done = env.step(action)

            # To mark boundaries between episodes
            if done:
                reward = 0

            acc_rewards.append(reward)
            acc_actions.append(action)

        # Update policy
        if episode > 0 and episode % batch_size == 0:
            # TODO
            pass
