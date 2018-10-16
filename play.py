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
        input_words = []
        action = SAY_HI

        while True:
            state, reward, done = env.step(action, state)
            
            enc_hidden = INITIAL_ENC_HIDDEN
            for w in state:
                _, enc_hidden = encoder(w, enc_hidden)
            dec_hidden = enc_hidden
            outputs = []
            the_word = START
            while w != EOS:
                # maybe apply softmax because w_prob may not be softmax-ed?
                w_probs, dec_hidden = decoder(the_word, dec_hidden)
                w_probs = softmax(w_probs)
                dist = Categorical(w_probs)
                the_word = dist.sample(w_probs)
                outputs.append(the_word)
            
            # action is a sentence (string)
            action = outputs.join('')

            # To mark boundaries between episodes
            # TODO: this part is copied from the cart pole example.
            # check if still necessary.
            if done:
                reward = 0
                input_words = []

            acc_rewards.append(reward)
            acc_actions.append(action)
            acc_states.append(state)

        # Update policy
        if episode > 0 and episode % BATCH_SIZE == 0:
            # TODO: this reward accumulation comes from the cartpole example.
            # may not be correct for our purpose. 
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

            optimizer = tf.train.RMSPropOptimizer(learning_rate=0.001)

            with tf.GradientTape() as tape:
                # accumulate gradient with GradientTape
                for i in range(steps):
                    # state is just the last sentence from user/environment
                    state = acc_states[i]
                    reward = acc_rewards[i]
                    action = acc_actions[i]

                    enc_hidden = INITIAL_ENC_HIDDEN
                    
                    for w in state:
                        _, enc_hidden = encoder(w, enc_hidden)
                    dec_hidden = enc_hidden
                    outputs = []
                    the_word = START
                    while w != EOS:
                        # maybe apply softmax because w_prob may not be softmax-ed?
                        w_probs, dec_hidden = decoder(the_word, dec_hidden)
                        w_probs = softmax(w_probs)
                        dist = Categorical(w_probs)
                        the_word = dist.sample(w_probs)
                        outputs.append(the_word)

                    dist = Categorical(probs)
                    # TODO: check formulation
                    loss = - log(dist(action)) * reward
                
                # calculate cumulative gradients
                model_vars = agent.get_model_variables()
                grads = tape.gradient(loss_value, model_vars)

                # this may be the place if we want to experiment with variable learning rates
                # grads = grads * lr
                
                # finally, apply gradient
                optimizer.apply_gradients(
                    zip(grads, model_vars),
                )

            acc_actions = []
            acc_rewards = []
            acc_state = []

