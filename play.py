# import pudb

# pudb.set_trace()

# import matplotlib.pyplot as plt
import numpy as np
from itertools import count
# from encoder_decoder_simple import get_decoder, get_encoder
from encoder_decoder import Encoder, Decoder
from environment import Environment, char_tokenizer, BEGIN_TAG, END_TAG, CONVO_LEN
# from agent import Baseline
import data
import random
import torch
from transformer.Models import Transformer
from transformer.dataset import collate_fn
import transformer.Constants as Constants
from corpus_utils import tokenize_sentence
import tensorflow as tf

# https://github.com/gabrielgarza/openai-gym-policy-gradient/blob/master/policy_gradient.py
# https://github.com/yaserkl/RLSeq2Seq/blob/7e019e8e8c006f464fdc09e77169680425e83ad1/src/model.py#L348

USE_CUDA = True
EPISODES = 10000000
BATCH_SIZE = 32
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
# UNITS = 128


def main():

    device = torch.device("cuda:0" if USE_CUDA else "cpu")

    env = Environment()

    END_TAG_IDX = env.lang.word2idx[END_TAG]

    SAY_HI = "hello"

    targ_lang = env.lang

    vocab_inp_size = len(env.lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)

    print("vocab_inp_size", vocab_inp_size)
    print("vocab_tar_size", vocab_tar_size)

    model = Transformer(
        vocab_inp_size,
        vocab_tar_size,
        MAX_TARGET_LEN).to(device)

    # baseline = Baseline(UNITS)

    history = []

    l_optimizer = torch.optim.Adam(model.parameters())

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
                for token in tokenize_sentence(sentence)]

    for episode in range(EPISODES):

        # Start of Episode
        env.reset()

        # get first state from the env
        state, _, done = env.step(SAY_HI)

        while not done:

            src_seq = [env.lang.word2idx[token]
                       for token in tokenize_sentence(state)]
            src_seq, src_pos = collate_fn([src_seq])
            src_seq, src_pos = src_seq.to(device), src_pos.to(device)
            enc_output, *_ = model.encoder(src_seq, src_pos)
            actions_t = []
            actions = []
            actions_idx = []

            while len(actions) == 0 or actions[len(actions)-1] != END_TAG_IDX and len(actions) < MAX_TARGET_LEN:
                # construct new tgt_seq based on what's outputed so far
                if len(actions_t) == 0:
                    tgt_seq = [env.lang.word2idx[Constants.UNK_WORD]]
                else:
                    tgt_seq = actions_idx
                tgt_seq, tgt_pos = collate_fn([tgt_seq])
                tgt_seq, tgt_pos = tgt_seq.to(device), tgt_pos.to(device)
                # dec_output dims: [1, pos, hidden]
                dec_output, * \
                    _ = model.decoder(tgt_seq, tgt_pos, src_seq, enc_output)
                # pick last step
                dec_output = dec_output[:, -1, :]
                # w_logits dims: [1, vocab_size]
                w_logits = model.tgt_word_prj(dec_output)
                # w_probs dims: [1, vocab_size]
                w_probs = torch.nn.functional.softmax(w_logits, dim=1)
                w_dist = torch.distributions.categorical.Categorical(
                    probs=w_probs)
                w_idx_t = w_dist.sample()
                w_idx = w_idx_t[0].cpu().numpy()
                actions_t.append(w_idx_t)
                actions_idx.append(w_idx)
                actions.append(env.lang.idx2word[w_idx])

            # action is a sentence (string)
            action_str = ' '.join(actions)
            next_state, reward, done = env.step(action_str)
            # print(reward)
            history.append((state, actions_t, action_str, reward))
            state = next_state

            # record history (to be used for gradient updating after the episode is done)
        # End of Episode
        # Update policy
        while len(history) >= BATCH_SIZE:
            batch = history[:BATCH_SIZE]
            state_inp_b, action_inp_b, reward_b, ret_seq_b = zip(*[
                [
                    sentence_to_idxs(state),
                    actions_b,
                    reward,
                    get_returns(reward, MAX_TARGET_LEN)
                ]
                for state, actions_b, _, reward in batch
            ])
            action_inp_b = list(action_inp_b)
            action_inp_b = torch.tensor(action_inp_b).to(device).unsqueeze(-1)

            ret_mean = np.mean(ret_seq_b)
            ret_std = np.std(ret_seq_b)
            ret_seq_b = (ret_seq_b - ret_mean) / ret_std
            ret_seq_b = torch.tensor(ret_seq_b).to(device)

            loss = 0
            # loss_bl=0
            l_optimizer.zero_grad()
            # accumulate gradient with GradientTape
            src_seq, src_pos = collate_fn(list(state_inp_b))
            src_seq, src_pos = src_seq.to(device), src_pos.to(device)
            enc_output_b, *_ = model.encoder(src_seq, src_pos)
            max_sentence_len = action_inp_b.shape[1]
            tgt_seq = [[Constants.BOS] for i in range(BATCH_SIZE)]
            for t in range(max_sentence_len):
                # _b stands for batch
                prev_w_idx_b, tgt_pos = collate_fn(tgt_seq)
                prev_w_idx_b, tgt_pos = prev_w_idx_b.to(
                    device), tgt_pos.to(device)
                # dec_output_b dims: [batch, pos, hidden]
                dec_output_b, *_ = \
                    model.decoder(prev_w_idx_b, tgt_pos, src_seq, enc_output_b)
                # pick last step
                dec_output_b = dec_output_b[:, -1, :]
                # w_logits_b dims: [batch, vocab_size]
                w_logits_b = model.tgt_word_prj(dec_output_b)
                # w_probs dims: [batch, vocab_size]
                w_probs_b = torch.nn.functional.softmax(w_logits_b, dim=1)

                dist_b = torch.distributions.categorical.Categorical(
                    probs=w_probs_b)
                curr_w_idx_b = action_inp_b[:, t]

                log_probs_b = torch.transpose(
                    dist_b.log_prob(
                        torch.transpose(curr_w_idx_b, 0, 1)
                    ), 0, 1
                )

                # bl_val_b = baseline(tf.cast(dec_hidden_b, 'float32'))
                # delta_b = ret_b - bl_val_b

                # cost_b = -tf.math.multiply(log_probs_b, delta_b)
                # cost_b = -tf.math.multiply(log_probs_b, ret_b)
                ret_b = torch.reshape(
                    ret_seq_b[:, t], (BATCH_SIZE, 1)).to(device)
                # alternatively, use torch.mul() but it is overloaded. Might need to try log_probs_b*vec.expand_as(A)
                cost_b = - log_probs_b.double() * ret_b.double()
                #  log_probs_b*vec.expand_as(A)
                # cost_b = -torch.bmm()   #if we are doing batch multiplication

                loss += cost_b
                # loss_bl += -tf.math.multiply(delta_b, bl_val_b)

                prev_w_idx_b = curr_w_idx_b
                tgt_seq = np.append(
                    tgt_seq, prev_w_idx_b.data.cpu().numpy(), axis=1).tolist()

            # calculate cumulative gradients

            # model_vars = encoder.variables + decoder.variables
            loss.sum().backward()
            # loss_bl.backward()

            # finally, apply gradient

            l_optimizer.step()
            # bl_optimizer.step()

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
            ret_seq_b_np = ret_seq_b.cpu().numpy()
            print(
                "all returns: min=%f, max=%f, median=%f" %
                (np.min(ret_seq_b_np),
                 np.max(ret_seq_b_np),
                 np.median(ret_seq_b_np))
            )
            print("avg reward: ", sum(reward_b) / len(reward_b))
            print("avg loss: ", np.mean(loss.cpu().numpy()))
            # print("avg grad: ", np.mean(grads[1].detach().cpu().numpy()))
            # print("<<<<<<<<<<<<<<<<<<<<<<<<<<")


main()
