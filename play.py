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
BATCH_SIZE = 64
# MODEL_BATCH_SIZE = 1
GAMMA = 1  # TODO
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
                      batch_sz=BATCH_SIZE, use_GloVe=USE_GLOVE, inp_lang=env.lang.vocab)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM, UNITS,
                      batch_sz=BATCH_SIZE, use_GloVe=USE_GLOVE, targ_lang=targ_lang.vocab)

    baseline = Baseline(UNITS)

    history = []

    l_optimizer = tf.train.AdamOptimizer()
    # bl_optimizer = tf.train.RMSPropOptimizer(0.01)

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
        if len(history) >= BATCH_SIZE:

            def get_returns(r: float, seq_len: int):
                return list(reversed([
                    r * (GAMMA ** t) for t in range(seq_len)
                ]))

            batch = history[:BATCH_SIZE]
            print("==========================")
            print("Episode # ", episode)
            print("Samples from episode with rewards > 0: ")
            good_rewards = [(s, a, r) for s, a, r in batch if r > 0]
            for s, a, r in random.sample(good_rewards, min(len(good_rewards), 5)):
                print("prev_state: ", s)
                print("action: ", a)
                print("reward: ", r)
                # print("return: ", get_returns(r, MAX_TARGET_LEN))

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

            state_inp_b, action_encs_b, reward_b, ret_seq_b = zip(*[
                [
                    sentence_to_idxs(state),
                    action_to_encs(action),
                    reward,
                    get_returns(reward, MAX_TARGET_LEN)
                ]
                for state, action, reward in batch
            ])
            action_encs_b = maybe_pad_sentence(action_encs_b)
            action_encs_b = tf.expand_dims(
                tf.convert_to_tensor(action_encs_b), -1)

            ret_mean = np.mean(ret_seq_b)
            ret_std = np.std(ret_seq_b)
            ret_seq_b = (ret_seq_b - ret_mean) / ret_std

            ret_seq_b = tf.cast(tf.convert_to_tensor(ret_seq_b), 'float32')

            print(
                "all returns: min=%f, max=%f, median=%f" %
                (np.min(ret_seq_b), np.max(ret_seq_b), np.median(ret_seq_b))
            )
            print("avg reward: ", sum(reward_b) / len(reward_b))

            loss = 0
            # loss_bl = 0

            with tf.GradientTape() as l_tape, tf.GradientTape() as bl_tape:
                # accumulate gradient with GradientTape
                init_hidden_b = initialize_hidden_state(BATCH_SIZE, UNITS)

                state_inp_b = maybe_pad_sentence(state_inp_b)
                state_inp_b = tf.convert_to_tensor(state_inp_b)

                enc_hidden_b = encoder(state_inp_b, init_hidden_b)
                dec_hidden_b = enc_hidden_b
                max_sentence_len = action_encs_b.shape[1]
                prev_w_idx_b = tf.expand_dims(
                    tf.cast(
                        tf.convert_to_tensor(
                            [env.lang.word2idx[BEGIN_TAG]] * BATCH_SIZE),
                        'float32'
                    ), -1
                )
                for t in range(max_sentence_len):

                    # bl_val_b = baseline(tf.cast(dec_hidden_b, 'float32'))
                    ret_b = tf.reshape(ret_seq_b[:, t], (BATCH_SIZE, 1))
                    # delta_b = ret_b - bl_val_b

                    w_probs_b, dec_hidden_b = decoder(
                        prev_w_idx_b, dec_hidden_b
                    )
                    curr_w_idx_b = action_encs_b[:, t]
                    dist = tf.distributions.Categorical(probs=w_probs_b)
                    # loss_bl += - \
                    #     tf.math.multiply(delta_b, bl_val_b)
                    # loss += tf.math.multiply(
                    #     tf.transpose(dist.log_prob(
                    #         tf.transpose(curr_w_idx_b))), delta_b
                    # )
                    cost_b = -tf.math.multiply(
                        tf.transpose(dist.log_prob(
                            tf.transpose(curr_w_idx_b))), ret_b
                    )
                    # print(cost_b.shape)
                    loss += cost_b

                    prev_w_idx_b = curr_w_idx_b

            print("avg loss: ", tf.reduce_mean(loss).numpy())

            # calculate cumulative gradients

            model_vars = encoder.variables + decoder.variables
            grads = l_tape.gradient(loss, model_vars)
            print("avg grad: ", np.mean(grads[1].numpy()))
            # grads_bl = bl_tape.gradient(loss_bl,  baseline.variables)

            # finally, apply gradient
            l_optimizer.apply_gradients(zip(grads, model_vars))
            # bl_optimizer.apply_gradients(zip(grads_bl, baseline.variables))

            # Reset everything for the next episode
            history = history[BATCH_SIZE:]


main()
