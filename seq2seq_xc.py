import tensorflow as tf
from tensorflow.keras.layers import (
    LSTMCell, LSTM, Dense, Embedding, RNN
)
from tensorflow.keras import(
    Sequential
)
from environment import Environment, char_tokenizer, BEGIN_TAG, END_TAG, CONVO_LEN


class Encoder(tf.keras.Model):
    def __init__(self, num_units, backwards, batch_size, embedding_dim, max_seq_len, src_vocab_size):
        super(tf.keras.Model, self).__init__()
        self.embd = Embedding(
            src_vocab_size, embedding_dim,
            batch_input_shape=[batch_size, None])
        self.lstm1 = LSTM(
            num_units,
            return_sequences=False,
            return_state=True,
            # go_backwards=backwards
        )

    def call(self, x_seq):
        x = self.embd(x_seq)
        _, h, c = self.lstm1(x)
        x = (h, c)
        return x


def get_decoder_cell(num_units, max_seq_len, targ_vocab_size):
    return Sequential([
        LSTMCell(
            num_units, input_shape=((1, targ_vocab_size)),
        ),
        Dense(targ_vocab_size, activation=None)
    ])


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


# def train(encoder: Encoder, decoder: Decoder):
#     x, y, sl, sos, w2i, i2w, i2e = get_data()
#     optimzer = tf.train.AdamOptimizer()

#     for _ in range(NUM_EPOCHS):
#         for x_batch, y_batch, sl_batch in zip(x, y, sl):
#             optimzer.minimize(lambda: get_loss(
#                 encoder, decoder, x_batch, y_batch, sl_batch, sos)
#             )


def cost_function(output, target, sl):
    cross_entropy = target * tf.log(tf.clip_by_value(output, 1e-10, 1.0))
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)
    mask = tf.cast(tf.sequence_mask(sl, output.shape[1]), dtype=tf.float32)
    cross_entropy *= mask

    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)


def get_loss(encoder: tf.keras.Model, decoder: tf.keras.Model, x, y, sl, sos):
    cell_state = encoder(x)

    _, wl = decoder(x, sos, cell_state, traning=True)
    loss = cost_function(wl, y, sl)
    return loss


from data.twitter.data import load_data, split_dataset
import copy


def remove_pad_sequences(sequences, pad_id=0):
    sequences_out = copy.deepcopy(sequences)

    for i, _ in enumerate(sequences):
        # for j in range(len(sequences[i])):
        #     if sequences[i][j] == pad_id:
        #         sequences_out[i] = sequences_out[i][:j]
        #         break
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
        max_seq_len=MAX_TOKENS_SRC,
        src_vocab_size=len(word2idx)
    )
    # decoder = get_decoder_cell(
    #     NUM_UNITS, MAX_TOKENS_TARG, len(word2idx)
    # )
    BUFFER_SIZE = len(trainX)

    dataset = list(zip(batch(trainX, BATCH_SIZE), batch(trainY, BATCH_SIZE)))

    for _ in range(NUM_EPOCHS):
        for batch, (x_batch, y_batch) in enumerate(dataset):
            x_batch = maybe_pad_sentences(x_batch)
            y_batch = maybe_pad_sentences(y_batch)
            encoder(tf.convert_to_tensor(x_batch, dtype='float32'))
