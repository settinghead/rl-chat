# import pudb

# pudb.set_trace()

# import matplotlib.pyplot as plt
import numpy as np
from itertools import count
# from encoder_decoder_simple import get_decoder, get_encoder
from encoder_decoder import Encoder, Decoder
from environment import Environment, char_tokenizer, BEGIN_TAG, END_TAG, CONVO_LEN
from agent import Baseline
import data
import random
import torch
import tf

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
    EMBEDDING_DIM = 8

MAX_TARGET_LEN = 20  # TODO: hack
UNITS = 128


def initialize_hidden_state(batch_sz, num_enc_units):
    return (
        tf.zeros((batch_sz, num_enc_units)),
        tf.zeros((batch_sz, num_enc_units)),
        tf.zeros((batch_sz, num_enc_units))
    )


def main():

    env = Environment()

    SAY_HI = "hello"

    targ_lang = env.lang

    vocab_inp_size = len(env.lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)

    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM,
                      UNITS, batch_sz=BATCH_SIZE, inp_lang=env.lang.vocab)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM,
                      UNITS, batch_sz=BATCH_SIZE, targ_lang=targ_lang.vocab)

    baseline = Baseline(UNITS)

    history = []

    l_optimizer = torch.optm.Adam()
    bl_optimizer = torch.optm.Adam()

    batch = None

    def maybe_pad_sentence(s):
        return tf.keras.preprocessing.sequence.pad_sequences(
            s,
            maxlen=MAX_TARGET_LEN,
            padding='post'
        )

    def get_returns(r: float, seq_len: int):
        return list(reversed([
            r * (GAMMA ** t) for t in range(seq_len)
        ]))

    def sentence_to_idxs(sentence: str):
        return [env.lang.word2idx[token]
                for token in char_tokenizer(sentence)]

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
            state_inp_b = maybe_pad_sentence([state_inp])

            # TODO: THIS IS NOT WORKING
            enc_hidden = encoder(
                torch.tensor(state_inp_b), init_hidden
            )
            dec_hidden = enc_hidden
            # TODO: THIS IS NOT WORKING

            w = BEGIN_TAG
            curr_w_enc = torch.unsqueeze(
                [targ_lang.word2idx[w]], 0
            )

            outputs = []
            actions = []
            while w != END_TAG and len(outputs) < MAX_TARGET_LEN:
                w_probs_b, dec_hidden = decoder(curr_w_enc, dec_hidden)
                w_dist = torch.distributions.categorical.Categorical(
                    probs=w_probs_b[0])
                w_idx = w_dist.sample(1)
                actions.append(w_idx)
                # w_idx = tf.argmax(w_probs[0]).numpy()[0]
                w = targ_lang.idx2word[w_idx.numpy()[0]]
                curr_w_enc = torch.unsqueeze(
                    [targ_lang.word2idx[w]] * 1, 1
                )
                outputs.append(w)

            # action is a sentence (string)
            action_str = ''.join(outputs)
            next_state, reward, done = env.step(action_str)
            history.append((state, actions, action_str, reward))
            state = next_state

            # record history (to be used for gradient updating after the episode is done)
        # End of Episode
        # Update policy
        while len(history) >= BATCH_SIZE:
            batch = history[:BATCH_SIZE]

            state_inp_b, action_encs_b, reward_b, ret_seq_b = zip(*[
                [
                    sentence_to_idxs(state),
                    actions_enc_b,
                    reward,
                    get_returns(reward, MAX_TARGET_LEN)
                ]
                for state, actions_enc_b, _, reward in batch
            ])
            action_encs_b = list(action_encs_b)
            action_encs_b = maybe_pad_sentence(action_encs_b)
            action_encs_b = torch.unsqueeze(
                torch.tensor(action_encs_b), -1)

            ret_mean = np.mean(ret_seq_b)
            ret_std = np.std(ret_seq_b)
            ret_seq_b = (ret_seq_b - ret_mean) / ret_std
            ret_seq_b = torch.tensor(ret_seq_b), dtype = torch.float32)

            loss=0
            loss_bl=0

            l_optimizer.zero_grad()
            # bl_optimizer.zero_grad()

            # with tf.GradientTape() as l_tape, tf.GradientTape() as bl_tape:
                # accumulate gradient with GradientTape
                init_hidden_b=initialize_hidden_state(BATCH_SIZE, UNITS)

                state_inp_b=maybe_pad_sentence(state_inp_b)
                state_inp_b=torch.tensor(state_inp_b)

                # enc_hidden_b=encoder(state_inp_b, init_hidden_b)
                src_enc, *_=self.model.encoder(src_seq, src_pos)

                dec_hidden_b=enc_hidden_b
                max_sentence_len=action_encs_b.shape[1]
                prev_w_idx_b=torch.unsqueeze(
                    torch.tensor(
                        [env.lang.word2idx[BEGIN_TAG]] * BATCH_SIZE,
                        torch.float32
                    ), -1
                )
                for t in range(max_sentence_len):

                    ret_b = tf.reshape(ret_seq_b[:, t], (BATCH_SIZE, 1))

                    # TODO: THIS IS NOT WORKING
                    dec_output, *_ = self.model.decoder(dec_seq, dec_pos, src_seq, enc_output)
                                    dec_output = dec_output[:, -1, :]  # Pick the last step: (bh * bm) * d_h
                                    word_prob = F.log_softmax(self.model.tgt_word_prj(dec_output), dim=1)
                                    word_prob = word_prob.view(n_active_inst, n_bm, -1)
                    # TODO: THIS IS NOT WORKING


                    # w_probs_b, dec_hidden_b = decoder(
                    #     prev_w_idx_b, dec_hidden_b
                    # )
                    curr_w_idx_b = action_encs_b[:, t]
                    # w_probs_b = tf.nn.softmax(w_logits_b)
                    dist = tf.distributions.Categorical(probs=w_probs_b)
                    log_probs_b = tf.transpose(
                        dist.log_prob(tf.transpose(curr_w_idx_b))
                    )
                    bl_val_b = baseline(tf.cast(dec_hidden_b, 'float32'))
                    delta_b = ret_b - bl_val_b

                    cost_b = -tf.math.multiply(log_probs_b, delta_b)
                    # cost_b = -tf.math.multiply(log_probs_b, ret_b)

                    loss += cost_b
                    loss_bl += -tf.math.multiply(delta_b, bl_val_b)

                    prev_w_idx_b = curr_w_idx_b

            # calculate cumulative gradients

            # model_vars = encoder.variables + decoder.variables
            loss.backward()
            loss_bl.backward()

            # finally, apply gradient

            l_optimizer.step()
            bl_optimizer.step()

            # Reset everything for the next episode
            history = history[BATCH_SIZE:]

        if episode % max(BATCH_SIZE, 128) == 0 and batch != None:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Episode # ", episode)
            print("Samples from episode with rewards > 0: ")
            good_rewards = [(s, a_str, r) for s, _, a_str, r in batch]
            for s, a, r in random.sample(good_rewards, min(len(good_rewards), 3)):
                print("prev_state: ", s)
                print("actions: ", a)
                print("reward: ", r)
                # print("return: ", get_returns(r, MAX_TARGET_LEN))
            print(
                "all returns: min=%f, max=%f, median=%f" %
                (np.min(ret_seq_b), np.max(ret_seq_b), np.median(ret_seq_b))
            )
            print("avg reward: ", sum(reward_b) / len(reward_b))
            print("avg loss: ", tf.reduce_mean(loss).numpy())
            print("avg grad: ", np.mean(grads[1].numpy()))
            # print("<<<<<<<<<<<<<<<<<<<<<<<<<<")


main()
