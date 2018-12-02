import tensorflow as tf
from tensorflow.keras.layers import (
    CuDNNLSTM, Dense, Embedding, RNN, LSTM
)
from tensorflow.keras import(
    Sequential
)
from environment import Environment, char_tokenizer, BEGIN_TAG, END_TAG, CONVO_LEN
from tqdm import tqdm
from tensorflow.contrib.seq2seq import (
    sequence_loss
)


class Encoder(tf.keras.Model):
    def __init__(self, num_units, backwards, batch_size, embedding_dim, src_vocab_size):
        super(tf.keras.Model, self).__init__()
        self.embd = Embedding(
            src_vocab_size, embedding_dim,
            batch_input_shape=[batch_size, None])
        self.lstm1 = CuDNNLSTM(
            num_units,
            return_sequences=False,
            return_state=True,
            # go_backwards=backwards
        )

    def call(self, x_seq):
        x = self.embd(x_seq)
        _, c, m = self.lstm1(x)
        return (c, m)


class DecoderCell(tf.keras.Model):
    def __init__(self, num_units, batch_size, embedding_dim, targ_vocab_size):
        super(tf.keras.Model, self).__init__()
        self.embd = Embedding(
            targ_vocab_size, embedding_dim,
            batch_input_shape=[batch_size, None])
        self.lstm_cell1 = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
            num_units
        )
        self.fc = Dense(targ_vocab_size, activation=None)

    def call(self, y_at_t, cell_state):
        x = self.embd(y_at_t)
        # print(x.shape, cell_state[0].shape)
        # print(y_at_t.shape, x.shape, cell_state[0].shape, cell_state[1].shape)
        o, h = self.lstm_cell1(x, cell_state)
        logits = self.fc(o)
        return logits, h


import numpy as np


def save(model: tf.keras.Model, optimizer, folder: str):
    saver = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model,
        optimizer_step=tf.train.get_or_create_global_step()
    )
    saver.save(folder)


def load(model: tf.keras.Model, optimizer,
         folder: str, bs: int, seq_len: int, hidden_dim: int):
    model(np.zeros((bs, seq_len, hidden_dim),
                   dtype=np.float32), list(range(2, bs + 2, 1)))
    saver = tf.train.Checkpoint(
        optimizer=optimizer,
        model=model,
        optimizer_step=tf.train.get_or_create_global_step()
    )
    saver.restore(folder)


NUM_EPOCHS = 300


def cost_function(output, target, sl):
    cross_entropy = target * tf.log(tf.clip_by_value(output, 1e-10, 1.0))
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)
    mask = tf.cast(tf.sequence_mask(sl, output.shape[1]), dtype=tf.float32)
    cross_entropy *= mask

    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)


from data.twitter.data import load_data, split_dataset
import copy


def remove_pad_sequences(sequences, pad_id=0):
    sequences_out = copy.deepcopy(sequences)

    for i, _ in enumerate(sequences):
        for j in range(1, len(sequences[i])):
            if sequences[i][-j] != pad_id:
                sequences_out[i] = sequences_out[i][0:-j + 1]
                break

    return sequences_out


def maybe_pad_sentences(s):
    return tf.keras.preprocessing.sequence.pad_sequences(
        s,
        padding='post',
        value=0  # TODO: change to symbolic
    )


def initial_setup():
    metadata, idx_q, idx_a = load_data(PATH='data/twitter/')
    (trainX, trainY), (testX, testY), (validX, validY) = split_dataset(idx_q, idx_a)
    trainX = remove_pad_sequences(trainX.tolist())
    trainY = remove_pad_sequences(trainY.tolist())
    testX = remove_pad_sequences(testX.tolist())
    testY = remove_pad_sequences(testY.tolist())
    validX = remove_pad_sequences(validX.tolist())
    validY = remove_pad_sequences(validY.tolist())
    return metadata, trainX, trainY, testX, testY, validX, validY


BATCH_SIZE = 32
NUM_UNITS = 128
MAX_TOKENS_SRC = 20
MAX_TOKENS_TARG = 20
EMBEDDING_DIM = 64


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


import random

if __name__ == '__main__':
    tf.enable_eager_execution()
    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup()

    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    n_step = src_len // BATCH_SIZE
    src_vocab_size = len(metadata['idx2w'])  # 8002 (0~8001)

    word2idx = metadata['w2idx']   # dict  word 2 index
    idx2word = metadata['idx2w']   # list index 2 word
    optimzer = tf.train.AdamOptimizer()

    encoder = Encoder(
        NUM_UNITS,
        backwards=False,
        batch_size=BATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        src_vocab_size=len(word2idx)
    )
    decoder_cell = DecoderCell(
        NUM_UNITS,
        batch_size=BATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        targ_vocab_size=len(word2idx)
    )

    dataset = list(zip(batch(trainX, BATCH_SIZE), batch(trainY, BATCH_SIZE)))
    for i in range(NUM_EPOCHS):
        for batch, (x_batch, y_batch) in enumerate(dataset):
            if(len(x_batch) != BATCH_SIZE):
                continue
            with tf.GradientTape() as tape:

                x_batch = maybe_pad_sentences(x_batch)
                sl_b = [len(y) for y in y_batch]
                y_batch = maybe_pad_sentences(y_batch)
                cell_state_b = encoder(
                    tf.convert_to_tensor(x_batch, dtype='float32')
                )

                o = tf.convert_to_tensor([0] * BATCH_SIZE, dtype='float32')
                logits_seq = []
                for idx in range(y_batch.shape[1] + 1):
                    # use teacher forcing
                    w_logits, cell_state_b = decoder_cell(o, cell_state_b)
                    if(idx > 0):
                        logits_seq.append(w_logits)

                    if (idx < y_batch.shape[1]):
                        o = tf.convert_to_tensor(y_batch[:, idx])

                logits_seq = tf.transpose(
                    tf.convert_to_tensor(logits_seq), [1, 0, 2]
                )
                masks = tf.cast(tf.sequence_mask(
                    sl_b, y_batch.shape[1]
                ), dtype='float32')
                targets = tf.convert_to_tensor(y_batch)
                loss = sequence_loss(
                    logits=logits_seq, targets=targets,
                    weights=masks
                )
                model_vars = encoder.variables + decoder_cell.variables
                grads = tape.gradient(loss, model_vars)
            optimzer.apply_gradients(zip(grads, model_vars))

            if batch % 40 == 0:
                sample_xs = random.sample(trainX, 32)
                sample_xs = maybe_pad_sentences(sample_xs)
                cell_state_b = encoder(
                    tf.convert_to_tensor(sample_xs, dtype='float32')
                )
                outputs = []
                for idx in range(MAX_TOKENS_TARG):
                    # use teacher forcing
                    w_logits, cell_state_b = decoder_cell(o, cell_state_b)
                    o = tf.nn.softmax(w_logits)
                    o = tf.math.argmax(o, axis=-1)
                    outputs.append(o.numpy())
                outputs = np.rollaxis(np.asarray(outputs), 0, 1)
                outputs = [
                    ' '.join([idx2word[w_idx] for w_idx in sentence])
                    for sentence in outputs
                ]
                print(random.sample(outputs, 5))

                print("loss: ", loss.numpy())
