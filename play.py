import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from encoder_decoder import Encoder, Decoder
from environment import Environment

tf.enable_eager_execution()

EPISODES = 100
BATCH_SIZE = 20
GAMMA = 0.9  # TODO


class Agent:
    def __init__(self):
        pass

    def encoder_z(self, state):
        pass

     def get_model_variables(self):
         pass

def main():
    env = Environment()
    agent = Agent()

    for episode in range(EPISODES):

        state = env.reset()

        acc_rewards = []
        acc_actions = []
        acc_states = []

        while True:
            encoder_z = agent.encoder_z(state)
            actions_with_probs = beam_search(encoder_z)
            actions_dist = Categorical(actions_with_probs)

            action = actions_dist.sample()


            state, reward, done = env.step(action)

            # To mark boundaries between episodes
            if done:
                reward = 0

            acc_rewards.append(reward)
            acc_actions.append(action)
            acc_states.append(state)

        # Update policy
        if episode > 0 and episode % BATCH_SIZE == 0:
            # TODO
            running_add = 0
            for i in reversed(range(steps)):
                if acc_rewards[i] == 0
                    running_add = 0
                else:
                    running_add = running add * GAMMA + acc_rewards[i]
                    acc_rewards[i] = running_add

            # normalize reward

            reward_mean = np.mean(acc_rewards)
            reward_std = np.std(acc_rewards)

            norm_rewards = [(r - reward_mean) / reward_std for r in acc_rewards]

            # accumulate gradient

            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

            with tf.GradientTape() as tape:
                for i in range(steps):
                    state = acc_states[i]
                    reward = acc_rewards[i]

                    encoder_z = agent.encoder_z(state)
                    actions_with_probs = beam_search(encoder_z)
                    actions_dist = Categorical(actions_with_probs)
                    loss = - log(actions_dist(action)) * reward
                
                model_vars = agent.get_model_variables()
                grads = tape.gradient(loss_value, model_vars)
                optimizer.apply_gradients(
                    zip(grads, model_vars),
                )

            acc_actions = []
            acc_rewards = []
            acc_state = []

