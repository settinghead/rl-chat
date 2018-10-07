import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from policy_net import Encoder, Decoder
from environment import Environment

tf.enable_eager_execution()

EPISODES = 100


def main():
    env = Environment()

    for episode in range(EPISODES):

        state = env.reset()

        for c in count():
            action = agent.calc(state)
            state, reward, done = env.step(action)

            # To mark boundaries between episodes
            if done:
                reward = 0
