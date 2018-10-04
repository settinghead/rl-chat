import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from policy_net import Encoder, Decoder

tf.enable_eager_execution()


class Environment:
    def step(self, action):
        pass


class Agent():
    def __init__(self):
        self.encoder = Encoder()
        self.decoder = Decoder()

    def calc(self, state):
        pass


EPOCHS = 100
INITIAL_STATE = []


def main():
    env = Environment()
    agent = Agent()

    for epoch in range(EPOCHS):

        for c in count():
            state = INITIAL_STATE
            action = agent.calc(state)
            state, reward, done = env.step(action)
