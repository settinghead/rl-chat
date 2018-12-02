import tensorflow as tf
#import matplotlib.pyplot as plt
tf.enable_eager_execution()

import pdb

import numpy as np
from itertools import count
from encoder_decoder import Encoder, Decoder
from environment_CoreNLP import Environment, char_tokenizer, BEGIN_TAG, END_TAG, CONVO_LEN
from agent import Baseline
import data
import random

from utils import load_trained_model, max_length
from embedding_utils import get_embedding_dim, get_GloVe_embeddings
from sklearn.metrics.pairwise import cosine_similarity


# https://github.com/gabrielgarza/openai-gym-policy-gradient/blob/master/policy_gradient.py
# https://github.com/yaserkl/RLSeq2Seq/blob/7e019e8e8c006f464fdc09e77169680425e83ad1/src/model.py#L348

EPISODES = 10000
BATCH_SIZE = 64
GAMMA = 1
USE_GLOVE = True
if USE_GLOVE:
    # 1024 if using glove
    EMBEDDING_DIM = 100
else:
    # 256 if without pretrained embedding
    EMBEDDING_DIM = 32

UNITS = 512

MAX_TARGET_LEN = 20  # TODO: hack


def initialize_hidden_state(batch_sz, num_enc_units):
    return tf.zeros((batch_sz, num_enc_units))

def get_returns(r: float, seq_len: int):
            return list(reversed([
                r * (GAMMA ** t) for t in range(seq_len)
            ]))

def sentence_to_idxs(sentence: str):
    return [env.lang.word2idx[token]
            for token in char_tokenizer(sentence)]

def maybe_pad_sentence(s):
    return tf.keras.preprocessing.sequence.pad_sequences(
        s,
        maxlen=MAX_TARGET_LEN,
        padding='post'
    )


def main():
    env = Environment()
    # print(env.lang.word2idx)

    SAY_HI = "hello"

    targ_lang = env.lang

    vocab_inp_size = len(env.lang.word2idx)
    vocab_tar_size = len(targ_lang.word2idx)


    #GET WORD SCORES
    # sentimental_words = ["absolutely","abundant","accept","acclaimed","accomplishment","achievement","action","active","activist","acumen","adjust","admire","adopt","adorable","adored","adventure","affirmation","affirmative","affluent","agree","airy","alive","alliance","ally","alter","amaze","amity","animated","answer","appreciation","approve","aptitude","artistic","assertive","astonish","astounding","astute","attractive","authentic","basic","beaming","beautiful","believe","benefactor","benefit","bighearted","blessed","bliss","bloom","bountiful","bounty","brave","bright","brilliant","bubbly","bunch","burgeon","calm","care","celebrate","certain","change","character","charitable","charming","cheer","cherish","clarity","classy","clean","clever","closeness","commend","companionship","complete","comradeship","confident","connect","connected","constant","content","conviction","copious","core","coupled","courageous","creative","cuddle","cultivate","cure","curious","cute","dazzling","delight","direct","discover","distinguished","divine","donate","each","day","eager","earnest","easy","ecstasy","effervescent","efficient","effortless","electrifying","elegance","embrace","encompassing","encourage","endorse","energized","energy","enjoy","enormous","enthuse","enthusiastic","entirely","essence","established","esteem","everyday","everyone","excited","exciting","exhilarating","expand","explore","express","exquisite","exultant","faith","familiar","family","famous","feat","fit","flourish","fortunate","fortune","freedom","fresh","friendship","full","funny","gather","generous","genius","genuine","give","glad","glow","good","gorgeous","grace","graceful","gratitude","green","grin","group","grow","handsome","happy","harmony","healed","healing","healthful","healthy","heart","hearty","heavenly","helpful","here","highest","good","hold","holy","honest","honor","hug","i","affirm","i","allow","i","am","willing","i","am.","i","can","i","choose","i","create","i","follow","i","know","i","know,","without","a","doubt","i","make","i","realize","i","take","action","i","trust","idea","ideal","imaginative","increase","incredible","independent","ingenious","innate","innovate","inspire","instantaneous","instinct","intellectual","intelligence","intuitive","inventive","joined","jovial","joy","jubilation","keen","key","kind","kiss","knowledge","laugh","leader","learn","legendary","let","go","light","lively","love","loveliness","lucidity","lucrative","luminous","maintain","marvelous","master","meaningful","meditate","mend","metamorphosis","mind-blowing","miracle","mission","modify","motivate","moving","natural","nature","nourish","nourished","novel","now","nurture","nutritious","one","open","openhanded","optimistic","paradise","party","peace","perfect","phenomenon","pleasure","plenteous","plentiful","plenty","plethora","poise","polish","popular","positive","powerful","prepared","pretty","principle","productive","project","prominent","prosperous","protect","proud","purpose","quest","quick","quiet","ready","recognize","refinement","refresh","rejoice","rejuvenate","relax","reliance","rely","remarkable","renew","renowned","replenish","resolution","resound","resources","respect","restore","revere","revolutionize","rewarding","rich","robust","rousing","safe","secure","see","sensation","serenity","shift","shine","show","silence","simple","sincerity","smart","smile","smooth","solution","soul","sparkling","spirit","spirited","spiritual","splendid","spontaneous","still","stir","strong","style","success","sunny","support","sure","surprise","sustain","synchronized","team","thankful","therapeutic","thorough","thrilled","thrive","today","together","tranquil","transform","triumph","trust","truth","unity","unusual","unwavering","upbeat","value","vary","venerate","venture","very","vibrant","victory","vigorous","vision","visualize","vital","vivacious","voyage","wealthy","welcome","well","whole","wholesome","willing","wonder","wonderful","wondrous","xanadu","yes","yippee","young","youth","youthful","zeal","zest","zing","zip"]
    sentimental_words = ["good", "excellent", "well"]
    targ_lang_embd = get_GloVe_embeddings(targ_lang.vocab, EMBEDDING_DIM)
    sentimental_words_embd = get_GloVe_embeddings(
        sentimental_words, EMBEDDING_DIM)
    sim_scores = np.dot(sentimental_words_embd, np.transpose(targ_lang_embd))
    print(sim_scores.shape)
    
    
    
    #LOAD PRETRAINED MODEL HERE
    #For now...
    encoder = Encoder(vocab_inp_size, EMBEDDING_DIM,
                      UNITS, batch_sz=BATCH_SIZE, inp_lang=env.lang.vocab)
    decoder = Decoder(vocab_tar_size, EMBEDDING_DIM,
                      UNITS, batch_sz=BATCH_SIZE, targ_lang=targ_lang.vocab)


    baseline = Baseline(UNITS)

    history = []

    l_optimizer = tf.train.AdamOptimizer()
    bl_optimizer = tf.train.RMSPropOptimizer(0.01)
    batch = None

    
    

    for episode in range(EPISODES):
        # Start of Episode
        env.reset()

        # get first state from the env
        state, _, done = env.step(SAY_HI)

        while not done:  #NOT REALLY USING DONE (Conv_length=1)

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


            outputs = []
            actions = []
            words_score = 0
            while w != END_TAG and len(outputs) < MAX_TARGET_LEN:
                w_probs_b, dec_hidden = decoder(curr_w_enc, dec_hidden)
                w_dist = tf.distributions.Categorical(probs=w_probs_b[0])
                w_idx = w_dist.sample(1)
                #pdb.set_trace() ######################################################################################
                actions.append(w_idx)
                # w_idx = tf.argmax(w_probs[0]).numpy()
                w = targ_lang.idx2word[w_idx.numpy()[0]]
                #pdb.set_trace() ######################################################################################

                #NEW: accumulate score of words in full response
                words_score += np.max(sim_scores[1:, w_idx.numpy()[0]])

                curr_w_enc = tf.expand_dims(
                    [targ_lang.word2idx[w]] * 1, 1)
                outputs.append(w)

            # action is a sentence (string)
            action_str = ''.join(outputs)
            next_state, reward, done = env.step(action_str)
            history.append((state, actions, action_str, reward+words_score)) #Reward is sentence score + words score
            state = next_state

            #pdb.set_trace() ######################################################################################

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

            #pdb.set_trace() ######################################################################################

            action_encs_b = list(action_encs_b)
            action_encs_b = maybe_pad_sentence(action_encs_b)
            action_encs_b = tf.expand_dims(
                tf.convert_to_tensor(action_encs_b), -1)

            ret_mean = np.mean(ret_seq_b)
            ret_std = np.std(ret_seq_b)
            ret_seq_b = (ret_seq_b - ret_mean) / ret_std



            ret_seq_b = tf.cast(tf.convert_to_tensor(ret_seq_b), 'float32')

            loss = 0
            loss_bl = 0

            with tf.GradientTape() as l_tape, tf.GradientTape() as bl_tape:
                # accumulate gradient with GradientTape
                init_hidden_b = initialize_hidden_state(BATCH_SIZE, UNITS)

                state_inp_b = maybe_pad_sentence(state_inp_b)
                state_inp_b = tf.convert_to_tensor(state_inp_b)

                enc_hidden_b = encoder(state_inp_b, init_hidden_b)
                dec_hidden_b = enc_hidden_b
                max_sentence_len = action_encs_b.numpy().shape[1]
                prev_w_idx_b = tf.expand_dims(
                    tf.cast(
                        tf.convert_to_tensor(
                            [env.lang.word2idx[BEGIN_TAG]] * BATCH_SIZE),
                        'float32'
                    ), -1
                )

                #pdb.set_trace() ######################################################################################

                for t in range(max_sentence_len):

                    bl_val_b = baseline(tf.cast(dec_hidden_b, 'float32'))
                    ret_b = tf.reshape(ret_seq_b[:, t], (BATCH_SIZE, 1))
                    delta_b = ret_b - bl_val_b
                    # print(prev_w_idx_b.shape)
                    w_probs_b, dec_hidden_b = decoder(
                        prev_w_idx_b, dec_hidden_b
                    )
                    curr_w_idx_b = action_encs_b[:, t]
                    # w_probs_b = tf.nn.softmax(w_logits_b)
                    dist = tf.distributions.Categorical(probs=w_probs_b)
                    loss_bl += - \
                         tf.multiply(delta_b, bl_val_b)
                    # cost_b = -tf.multiply(
                    #     tf.transpose(dist.log_prob(
                    #         tf.transpose(curr_w_idx_b))), delta_b
                    # )

                    #pdb.set_trace() ######################################################################################
                    cost_b = -tf.multiply(
                        tf.transpose(dist.log_prob(
                            tf.transpose(curr_w_idx_b))), ret_b
                    )
                    # print(cost_b.shape)
                    loss += cost_b

                    prev_w_idx_b = curr_w_idx_b

                    #pdb.set_trace() ######################################################################################

            # calculate cumulative gradients

            pdb.set_trace() ######################################################################################

            model_vars = encoder.variables + decoder.variables
            grads = l_tape.gradient(loss, model_vars)

            grads_bl = bl_tape.gradient(loss_bl,  baseline.variables)

            # finally, apply gradient
            l_optimizer.apply_gradients(zip(grads, model_vars))
            bl_optimizer.apply_gradients(zip(grads_bl, baseline.variables))

            # Reset everything for the next episode
            history = history[BATCH_SIZE:]

        if episode % 20 == 0 and batch != None:
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("Episode # ", episode)
            print("Samples from episode with rewards > 0: ")
            good_rewards = [(s, a_str, r) for s, _, a_str, r in batch]
            for s, a, r in random.sample(good_rewards, min(len(good_rewards), 3)):
                print("prev_state: ", s)
                print("action: ", a)
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