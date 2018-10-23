import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from encoder_decoder import Encoder, Decoder, initialize_hidden_state
from environment import Environment
from corpus_utils import tokenize_sentence, LanguageIndex, BEGIN_TAG, END_TAG
import data
import random


EPISODES = 1000
BATCH_SIZE = 32
MODEL_BATCH_SIZE = 1
GAMMA = 1.0  # TODO
USE_GLOVE = True
if USE_GLOVE:
    # 1024 if using glove
    EMBEDDING_DIM = 100
else:
    # 256 if without pretrained embedding
    EMBEDDING_DIM = 128

MAX_TARGET_LEN = 20  # TODO: hack
UNITS = 128


def main():
    tf.enable_eager_execution()

    env = Environment()

    SAY_HI = "hello"

    targ_lang = env.lang

    vocab_inp_size = len(env.lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)

    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM, UNITS,
                      batch_sz=MODEL_BATCH_SIZE, use_GloVe=USE_GLOVE, inp_lang=env.lang.vocab)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS,
                      batch_sz=MODEL_BATCH_SIZE, use_GloVe=USE_GLOVE, targ_lang=targ_lang.vocab)

    history = []

    for episode in range(EPISODES):

        # Start of Episode
        state = env.reset()

        state, _, done = env.step(SAY_HI)

        while not done:
            # Assume initial hidden state is default, don't use: #enc_hidden = INITIAL_ENC_HIDDEN

            # Run an episode using the TRAINED ENCODER-DECODER model #TODO: test this!!
            init_hidden = initialize_hidden_state(MODEL_BATCH_SIZE, UNITS)
            state_inp = [env.lang.word2idx[token]
                         for token in tokenize_sentence(state)]
            enc_hidden = encoder(
                tf.convert_to_tensor([state_inp]), init_hidden)
            dec_hidden = enc_hidden

            w = BEGIN_TAG
            curr_w_enc = tf.expand_dims(
                [targ_lang.word2idx[w]] * MODEL_BATCH_SIZE, 1)

            outputs = []
            while w != END_TAG and len(outputs) < MAX_TARGET_LEN:
                w_probs, dec_hidden = decoder(curr_w_enc, dec_hidden)
                w_dist = tf.distributions.Categorical(w_probs)
                w_idx = w_dist.sample(1).numpy()[0][0]
                w = targ_lang.idx2word[w_idx]
                outputs.append(w)

            # action is a sentence (string)
            action = ' '.join(outputs)
            state, reward, done = env.step(action)

            # record history (to be used for gradient updating after the episode is done)
            history.append((action, state, reward))
        # End of Episode

        # Update policy
        if episode > 0 and episode % BATCH_SIZE == 0:

            for a, s, r in random.sample(history, 10):
                print("state: ", s)
                print("action: ", a)
                print("reward: ", r)

            # TODO: this reward accumulation comes from the cartpole example.
            # may not be correct for our purpose.
            running_add = 0

            acc_rewards = []

            for _, _, curr_reward in reversed(history):
                # if curr_reward == 0:
                #     running_add = 0
                # else:
                #     running_add = running_add * GAMMA + curr_reward
                acc_rewards.append(curr_reward)

            # normalize reward
            reward_mean = np.mean(acc_rewards)
            reward_std = np.std(acc_rewards)
            norm_rewards = [(r - reward_mean) /
                            reward_std for r in acc_rewards]
            print("all reward: ", norm_rewards)
            optimizer = tf.train.AdamOptimizer()

            with tf.GradientTape() as tape:
                # accumulate gradient with GradientTape
                for (action, state, _), norm_reward in zip(history, norm_rewards):
                    init_hidden = initialize_hidden_state(
                        MODEL_BATCH_SIZE, UNITS)
                    state_inp = [env.lang.word2idx[token]
                                 for token in tokenize_sentence(state)]
                    enc_hidden = encoder(
                        tf.convert_to_tensor([state_inp]), init_hidden)
                    dec_hidden = enc_hidden

                    begin_w = tf.expand_dims(
                        [targ_lang.word2idx[BEGIN_TAG]] * MODEL_BATCH_SIZE, 1)
                    action_words_enc = [begin_w] + [
                        tf.expand_dims([targ_lang.word2idx[token]]
                                       * MODEL_BATCH_SIZE, 1)
                        for token in tokenize_sentence(action)]

                    for curr_w_idx in action_words_enc:
                        w_probs, dec_hidden = decoder(curr_w_idx, dec_hidden)
                        # TODO: check if softmax is necessary
                        w_probs = tf.nn.softmax(w_probs)
                        dist = tf.distributions.Categorical(w_probs)
                        # TODO: check formulation
                        # TODO: determine if should add discount factor here
                        loss = - dist._log_prob(curr_w_idx) * norm_reward

                # calculate cumulative gradients
                model_vars = encoder.variables + decoder.variables
                grads = tape.gradient(loss, model_vars)
                # this may be the place if we want to experiment with variable learning rates
                # grads = grads * lr

            # finally, apply gradient
            optimizer.apply_gradients(
                zip(grads, model_vars),
            )

            # Reset everything for the next episode
            history = []


main()
