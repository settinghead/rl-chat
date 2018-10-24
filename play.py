import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from encoder_decoder import Encoder, Decoder, initialize_hidden_state
from environment import Environment, char_tokenizer, BEGIN_TAG, END_TAG
import data
import random

# https://github.com/gabrielgarza/openai-gym-policy-gradient/blob/master/policy_gradient.py
# https://github.com/yaserkl/RLSeq2Seq/blob/7e019e8e8c006f464fdc09e77169680425e83ad1/src/model.py#L348

EPISODES = 10000000
BATCH_SIZE = 64
MODEL_BATCH_SIZE = 1
GAMMA = 0.1  # TODO
USE_GLOVE = False
if USE_GLOVE:
    # 1024 if using glove
    EMBEDDING_DIM = 100
else:
    # 256 if without pretrained embedding
    EMBEDDING_DIM = 16

MAX_TARGET_LEN = 20  # TODO: hack
UNITS = 128


def main():
    tf.enable_eager_execution()

    env = Environment()
    # print(env.lang.word2idx)

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
        env.reset()

        # get first state from the env
        state, _, done = env.step(SAY_HI)

        while not done:
            encoder.reset_states()
            decoder.reset_states()
            # Assume initial hidden state is default, don't use: #enc_hidden = INITIAL_ENC_HIDDEN

            # Run an episode using the TRAINED ENCODER-DECODER model #TODO: test this!!
            init_hidden = initialize_hidden_state(MODEL_BATCH_SIZE, UNITS)
            state_inp = [env.lang.word2idx[token]
                         for token in char_tokenizer(state)]
            enc_hidden = encoder(
                tf.convert_to_tensor([state_inp]), init_hidden)
            dec_hidden = enc_hidden

            w = BEGIN_TAG
            curr_w_enc = tf.expand_dims(
                [targ_lang.word2idx[w]] * MODEL_BATCH_SIZE, 1)

            outputs = [w]
            while w != END_TAG and len(outputs) < MAX_TARGET_LEN:
                w_probs, dec_hidden = decoder(curr_w_enc, dec_hidden)
                w_dist = tf.distributions.Categorical(w_probs)
                w_idx = w_dist.sample(1).numpy()[0][0]
                w = targ_lang.idx2word[w_idx]
                curr_w_enc = tf.expand_dims(
                    [targ_lang.word2idx[w]] * MODEL_BATCH_SIZE, 1)
                outputs.append(w)
            # action is a sentence (string)
            action = ''.join(outputs)

            next_state, reward, done = env.step(action)
            history.append((state, action, reward))
            state = next_state

            # record history (to be used for gradient updating after the episode is done)
        # End of Episode

        # Update policy
        if episode > 0 and episode % BATCH_SIZE == 0:

            print("==========================")
            print("Episode # ", episode)
            print("Samples from episode: ")

            good_rewards = [(s, a, r) for s, a, r in history if r > 0]
            for s, a, r in random.sample(good_rewards, min(len(good_rewards), 5)):
                print("prev_state: ", s)
                print("action: ", a)
                print("reward: ", r)

            # TODO: this reward accumulation comes from the cartpole example.
            # may not be correct for our purpose.
            running_add = 0

            acc_rewards = []

            for _, _, curr_reward in reversed(history):
                if curr_reward == 0:
                    running_add = 0
                else:
                    running_add = running_add * GAMMA + curr_reward
                acc_rewards.append(curr_reward)
            acc_rewards = list(reversed(acc_rewards))

            # normalize reward
            reward_mean = np.mean(acc_rewards)
            reward_std = np.std(acc_rewards)
            norm_rewards = [(r - reward_mean) /
                            reward_std for r in acc_rewards]
            print("all reward: ", acc_rewards)
            print("avg rewards: ", sum(
                acc_rewards) / len(acc_rewards))
            optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
            loss = 0
            with tf.GradientTape() as tape:
                # accumulate gradient with GradientTape
                for (state, action, _), norm_reward in zip(history, norm_rewards):
                    encoder.reset_states()
                    decoder.reset_states()

                    init_hidden = initialize_hidden_state(
                        MODEL_BATCH_SIZE, UNITS)
                    state_inp = [env.lang.word2idx[token]
                                 for token in char_tokenizer(state)]
                    enc_hidden = encoder(
                        tf.convert_to_tensor([state_inp]), init_hidden)
                    dec_hidden = enc_hidden

                    begin_w = tf.expand_dims(
                        [targ_lang.word2idx[BEGIN_TAG]] * MODEL_BATCH_SIZE, 1)
                    action_words_enc = [begin_w] + [
                        tf.expand_dims([targ_lang.word2idx[token]]
                                       * MODEL_BATCH_SIZE, 1)
                        for token in char_tokenizer(action)]

                    for curr_w_idx in action_words_enc:
                        w_probs, dec_hidden = decoder(curr_w_idx, dec_hidden)
                        # TODO: check if softmax is necessary
                        # w_probs = tf.nn.softmax(w_probs)
                        dist = tf.distributions.Categorical(w_probs)
                        # TODO: check formulation
                        # TODO: determine if should add discount factor here
                        loss += - dist.log_prob(curr_w_idx) * norm_reward

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
