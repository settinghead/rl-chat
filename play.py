import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
from encoder_decoder import Encoder, Decoder, initialize_hidden_state
from environment import Environment, char_tokenizer, BEGIN_TAG, END_TAG, CONVO_LEN
from agent import Baseline
import data
import random

# https://github.com/gabrielgarza/openai-gym-policy-gradient/blob/master/policy_gradient.py
# https://github.com/yaserkl/RLSeq2Seq/blob/7e019e8e8c006f464fdc09e77169680425e83ad1/src/model.py#L348

EPISODES = 10000000
BATCH_SIZE = 128
# MODEL_BATCH_SIZE = 1
GAMMA = 0.7  # TODO
USE_GLOVE = False
if USE_GLOVE:
    # 1024 if using glove
    EMBEDDING_DIM = 100
else:
    # 256 if without pretrained embedding
    EMBEDDING_DIM = 5

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
                      batch_sz=BATCH_SIZE, use_GloVe=USE_GLOVE, inp_lang=env.lang.vocab)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS,
                      batch_sz=BATCH_SIZE, use_GloVe=USE_GLOVE, targ_lang=targ_lang.vocab)

    baseline = Baseline()

    history = []

    l_optimizer = tf.train.RMSPropOptimizer(0.01)
    bl_optimizer = tf.train.RMSPropOptimizer(0.01)

    for episode in range(EPISODES):

        # Start of Episode
        env.reset()

        # get first state from the env
        state, _, done = env.step(SAY_HI)

        while not done:
            # Assume initial hidden state is default, don't use: #enc_hidden = INITIAL_ENC_HIDDEN

            # Run an episode using the TRAINED ENCODER-DECODER model #TODO: test this!!
            init_hidden = initialize_hidden_state(1, UNITS)
            state_inp = [env.lang.word2idx[token]
                         for token in char_tokenizer(state)]
            enc_hidden = encoder(
                tf.convert_to_tensor([state_inp]), init_hidden)
            dec_hidden = enc_hidden

            w = BEGIN_TAG
            curr_w_enc = tf.expand_dims(
                [targ_lang.word2idx[w]], 0
            )

            outputs = [w]
            while w != END_TAG and len(outputs) < MAX_TARGET_LEN:
                w_probs, dec_hidden = decoder(curr_w_enc, dec_hidden)
                w_dist = tf.distributions.Categorical(w_probs[0])
                w_idx = w_dist.sample(1).numpy()[0]
                # w_idx = tf.argmax(w_probs[0]).numpy()
                w = targ_lang.idx2word[w_idx]
                curr_w_enc = tf.expand_dims(
                    [targ_lang.word2idx[w]] * 1, 1)
                outputs.append(w)
            # action is a sentence (string)
            action = ''.join(outputs)

            next_state, reward, done = env.step(action)
            history.append((state, action, reward))
            state = next_state

            # record history (to be used for gradient updating after the episode is done)
        # End of Episode
        # Update policy
        if episode > 0 and (episode + 1) % (BATCH_SIZE / CONVO_LEN * 2) == 0:
            print("==========================")
            print("Episode # ", episode)
            print("Samples from episode with rewards > 0: ")

            good_rewards = [(s, a, r) for s, a, r in history if r > 0]
            for s, a, r in random.sample(good_rewards, min(len(good_rewards), 5)):
                print("prev_state: ", s)
                print("action: ", a)
                print("reward: ", r)

            # TODO: this reward accumulation comes from the cartpole example.
            # may not be correct for our purpose.
            running_add = 0

            acc_reward_b = []

            for _, _, curr_reward in reversed(history):
                if curr_reward == 0:
                    running_add = 0
                else:
                    running_add = running_add * GAMMA + curr_reward
                acc_reward_b.append(running_add)
            acc_reward_b = list(reversed(acc_reward_b))

            # normalize reward
            reward_mean = np.mean(acc_reward_b)
            reward_std = np.std(acc_reward_b)
            norm_reward_b = [(r - reward_mean) /
                             reward_std for r in acc_reward_b]
            print(
                "all rewards: min=%f, max=%f, median=%f" %
                (np.min(acc_reward_b), np.max(acc_reward_b), np.median(acc_reward_b))
            )
            print("avg reward: ", sum(
                acc_reward_b) / len(acc_reward_b))

            loss = 0
            loss_bl = 0

            def sentence_to_idxs(sentence: str):
                return [env.lang.word2idx[token]
                        for token in char_tokenizer(sentence)]

            def action_to_encs(action: str):
                begin_w = tf.expand_dims(
                    [targ_lang.word2idx[BEGIN_TAG]], 1)
                enc = [begin_w] + [
                    tf.expand_dims([targ_lang.word2idx[token]], 1)
                    for token in char_tokenizer(action)
                ]
                return enc

            def maybe_pad_sentence(s):
                return tf.keras.preprocessing.sequence.pad_sequences(
                    s,
                    maxlen=MAX_TARGET_LEN,
                    padding='post'
                )

            with tf.GradientTape() as l_tape, tf.GradientTape() as bl_tape:
                # accumulate gradient with GradientTape
                init_hidden_b = initialize_hidden_state(BATCH_SIZE, UNITS)

                state_inp_b, action_encs_b = zip(*[
                    [sentence_to_idxs(state), action_to_encs(action)]
                    for (state, action, _), norm_reward in zip(history, norm_reward_b)
                ])
                state_inp_b = maybe_pad_sentence(state_inp_b)
                state_inp_b = tf.convert_to_tensor(state_inp_b)

                bl_val = baseline(tf.cast(state_inp_b, 'float32'))
                norm_reward_b = tf.cast(
                    tf.convert_to_tensor(norm_reward_b), 'float32')
                norm_reward_b -= bl_val

                action_encs_b = maybe_pad_sentence(action_encs_b)
                action_encs_b = tf.expand_dims(
                    tf.convert_to_tensor(action_encs_b), -1)
                enc_hidden_b = encoder(state_inp_b, init_hidden_b)
                dec_hidden_b = enc_hidden_b
                max_sentence_len = action_encs_b.shape[1]
                for i in range(max_sentence_len):
                    curr_w_idx_b = action_encs_b[:, i]
                    w_probs_b, dec_hidden_b = decoder(
                        curr_w_idx_b, dec_hidden_b)
                    # w_probs = tf.nn.softmax(w_probs)
                    dist = tf.distributions.Categorical(w_probs_b)
                    # TODO: check formulation
                    loss += - dist.log_prob(curr_w_idx_b) * norm_reward_b
                    loss_bl += norm_reward_b

            # calculate cumulative gradients
            model_vars = encoder.variables + decoder.variables
            grads = l_tape.gradient(loss, model_vars)
            grads_bl = bl_tape.gradient(loss_bl,  baseline.variables)
            # this may be the place if we want to experiment with variable learning rates
            # grads = grads * lr

            # finally, apply gradient
            l_optimizer.apply_gradients(zip(grads, model_vars))
            bl_optimizer.apply_gradients(zip(grads_bl, baseline.variables))

            # Reset everything for the next episode
            history = []


main()
