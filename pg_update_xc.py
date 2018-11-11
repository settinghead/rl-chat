import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
from itertools import count
import encoder_decoder
from corpus_utils import tokenize_sentence, LanguageIndex, BEGIN_TAG, END_TAG
from utils import load_trained_model, max_length, get_GloVe_embeddings
import data
import random
from sklearn.metrics.pairwise import cosine_similarity
import os
import time
from embedding_utils import get_embedding_dim

EPISODES = 1000
BATCH_SIZE = 64
TOP_K = 4
GAMMA = 1

USE_GLOVE = True
EMBEDDING_DIM = get_embedding_dim(USE_GLOVE)

UNITS = 512


def main():
    tf.enable_eager_execution()

    questions1, answers1 = data.load_conv_text()
    # questions2, answers2 = data.load_opensubtitles_text()

    questions = list(questions1)
    answers = list(answers1)

    inp_lang, targ_lang = LanguageIndex(questions), LanguageIndex(answers)

    input_tensor = [[inp_lang.word2idx[token]
                     for token in tokenize_sentence(question)] for question in questions]
    target_tensor = [[targ_lang.word2idx[token]
                      for token in tokenize_sentence(answer)] for answer in answers]
    max_length_inp, max_length_tar = max_length(
        input_tensor), max_length(target_tensor)
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=max_length_inp,
                                                                 padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=max_length_tar,
                                                                  padding='post')
    BUFFER_SIZE = len(input_tensor)
    dataset = tf.data.Dataset.from_tensor_slices(
        (input_tensor, target_tensor)).shuffle(BUFFER_SIZE)
    dataset = dataset.batch(BATCH_SIZE, drop_remainder=True)

    model: encoder_decoder.Seq2Seq = load_trained_model(
        BATCH_SIZE, EMBEDDING_DIM, UNITS, tf.train.AdamOptimizer())

    targ_lang_embd = get_GloVe_embeddings(targ_lang.vocab, EMBEDDING_DIM)

    #max_prob_ids = np.argmax(sim_scores, axis=1)
    # print(max_prob_ids)
    # print(targ_lang.word2idx)
    # print(targ_lang.idx2word(max_prob_ids[1]))

    optimizer = tf.train.AdamOptimizer()

    checkpoint_dir = './training_checkpoints'
    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, seq2seq=model)

    for episode in range(EPISODES):

        # Start of Episode
        start = time.time()
        total_loss = 0
        for (batch, (inp, targ)) in enumerate(dataset):
            with tf.GradientTape() as tape:

                hidden = tf.zeros((BATCH_SIZE, UNITS))
                enc_hidden = model.encoder(inp, hidden)
                dec_hidden = enc_hidden
                dec_input = tf.expand_dims(
                    [targ_lang.word2idx[BEGIN_TAG]] * BATCH_SIZE, 1)

                loss = 0  # loss for decoder
                pg_loss = 0  # loss for semantic

                result = ''
                for t in range(1, targ.shape[1]):
                    actions = []
                    probs = []
                    rewards = []
                    predictions, dec_hidden = model.decoder(
                        dec_input, dec_hidden)
                    # using teacher forcing
                    dec_input = tf.expand_dims(targ[:, t], 1)
                    for ps in predictions:
                        top_k_indices = tf.nn.top_k(ps, TOP_K).indices.numpy()
                        action = np.random.choice(top_k_indices, 1)[0]
                        actions.append(action)
                        prob = ps.numpy()[action]
                        probs.append(prob)
                        reward = np.mean(sim_scores[:, action])
                        # print(targ_lang.idx2word[action], reward)
                        rewards.append(reward)

                        # normalize reward
                        reward_mean = np.mean(rewards)
                        reward_std = np.std(rewards)
                        norm_rewards = [(r - reward_mean) /
                                        reward_std for r in rewards]

                    if targ_lang.idx2word[actions[0]] == END_TAG:
                        pass
                    else:
                        result += ' ' + targ_lang.idx2word[actions[0]]

                    onehot_labels = tf.keras.utils.to_categorical(
                        y=actions, num_classes=len(targ_lang.word2idx))

                    norm_rewards = tf.convert_to_tensor(
                        norm_rewards, dtype="float32") * GAMMA
                    # print(onehot_labels.shape)
                    # print(predictions.shape)
                    loss += model.loss_function(targ[:, t], predictions)
                    # print("------")
                    # print(loss)
                    # print(probs)
                    #pg_loss_cross = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=onehot_labels, logits=targ[:, t]))
                    pg_loss_cross = model.loss_function(
                        targ[:, t], onehot_labels)
                    pg_loss_cross = tf.reduce_mean(
                        pg_loss_cross * norm_rewards)
                    # print(pg_loss_cross)
                    # print("------")
                    # print(pg_loss_cross)
                    pg_loss += pg_loss_cross
                print("result: ", result)
                # End of Episode
                # Update policy
                batch_loss = ((loss + pg_loss) / int(targ.shape[1]))
                total_loss += batch_loss
                variables = model.encoder.variables + model.decoder.variables
                gradients = tape.gradient(loss, variables)
                optimizer.apply_gradients(zip(gradients, variables))
                if batch % 10 == 0:
                    print('batch {} training loss {:.4f}'.format(
                        batch, total_loss.numpy()))

        # saving (checkpoint) the model every 100 epochs
        if (episode + 1) % 100 == 0:
            checkpoint.save(file_prefix=checkpoint_prefix)

        print('Time taken for {} episode {} sec\n'.format(
            episode, time.time() - start))


main()
