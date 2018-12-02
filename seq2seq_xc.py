import tensorflow as tf
from tensorflow.keras.layers import (
    Dense, Embedding, RNN, LSTM, CuDNNLSTM, Dropout
)
from tensorflow.keras.activations import relu
from tensorflow.keras import(
    Sequential
)
from environment import Environment, char_tokenizer, BEGIN_TAG, END_TAG, CONVO_LEN
from tqdm import tqdm
from tensorflow.contrib.seq2seq import (
    sequence_loss
)

from tensorlayer.layers import DenseLayer, EmbeddingInputlayer, Seq2Seq, retrieve_seq_length_op2


class Encoder(tf.keras.Model):
    def __init__(self, num_units, backwards, batch_size, embedding_dim, src_vocab_size):
        super(tf.keras.Model, self).__init__()
        self.embd = Embedding(
            src_vocab_size, embedding_dim,
            batch_input_shape=[batch_size, None])
        self.lstm_layers = Sequential([
            # Dropout(0.5),
            CuDNNLSTM(
                num_units,
                return_sequences=True,
                return_state=False,
                go_backwards=backwards
            ),
            # Dropout(0.5),
            CuDNNLSTM(
                num_units,
                return_sequences=True,
                return_state=False
            ),
            # Dropout(0.5),
            CuDNNLSTM(
                num_units,
                return_sequences=False,
                return_state=True
            )
        ])
        self.dropout = Dropout(0.5)

    def call(self, x_seq, is_training):
        x = self.embd(x_seq)
        _, c, m = self.lstm_layers(x)
        return (c, m)


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
            go_backwards=backwards
        )
        keep_prob = 0.5
        self.dropout = Dropout(keep_prob)

    def call(self, x_seq, is_training):
        x = self.embd(x_seq)
        if is_training:
            x = tf.nn.dropout(x, 0.5)
        _, c, m = self.lstm1(x)
        return (c, m)


class DecoderCell(tf.keras.Model):
    def __init__(self, num_units, embd, batch_size, embedding_dim, targ_vocab_size):
        super(tf.keras.Model, self).__init__()
        self.embd = embd
        self.lstm_cell1 = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(
            num_units
        )
        self.fc = Dense(targ_vocab_size, activation=None)

    def call(self, y_at_t, cell_state, is_training):
        x = self.embd(y_at_t)
        if is_training:
            x = tf.nn.dropout(x, 0.5)
        # print(x.shape, cell_state[0].shape)
        # print(y_at_t.shape, x.shape, cell_state[0].shape, cell_state[1].shape)
        x, h = self.lstm_cell1(x, cell_state)
        if is_training:
            x = tf.nn.dropout(x, 0.5)
        logits = self.fc(x)
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


NUM_EPOCHS = 100


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


BATCH_SIZE = 128
NUM_UNITS = 128
MAX_TOKENS_SRC = 20
MAX_TOKENS_TARG = 20
EMBEDDING_DIM = 512


def batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


def decode_sentence(word_idxs, idx2word):
    return ' '.join([idx2word[w_idx] for w_idx in word_idxs if w_idx != 0])


import random
from data import load_conv_text, load_twitter_text
from corpus_utils import LanguageIndex, tokenize_sentence, EMPTY_IDX, UNKNOWN_IDX, filter_line


def encode_sentence(sentence, lang: LanguageIndex):
    return [lang.word2idx.get(w, lang.word2idx[lang._unknown_token]) for w in tokenize_sentence(sentence)]


if __name__ == '__main__':
    tf.enable_eager_execution()
    # metadata, trainX, trainY = load_data(PATH='data/twitter/')
    # why not both?
    questions1, answers1 = load_conv_text()
    questions2, answers2 = load_twitter_text()
    questions = list(questions1) + list(questions2)
    answers = list(answers1) + list(answers2)
    # questions, answers = data.load_conv_text()

    questions = [filter_line(q) for q in questions]
    answers = [filter_line(a) for a in answers]
    lang = LanguageIndex(questions + answers)

    trainX = [encode_sentence(s, lang) for s in questions]
    trainY = [encode_sentence(s, lang) for s in answers]
    print("Dataset size: ", len(trainX))

    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    print("Vocab size: ", len(lang.word2idx))

    optimizer = tf.train.AdamOptimizer()

    encoder = Encoder(
        NUM_UNITS,
        backwards=False,
        batch_size=BATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        src_vocab_size=len(lang.word2idx)
    )
    decoder_cell = DecoderCell(
        NUM_UNITS,
        encoder.embd,
        batch_size=BATCH_SIZE,
        embedding_dim=EMBEDDING_DIM,
        targ_vocab_size=len(lang.word2idx)
    )

    dataset = list(zip(batch(trainX, BATCH_SIZE), batch(trainY, BATCH_SIZE)))
    for epoch in range(NUM_EPOCHS):
        for batch, (x_batch, y_batch) in enumerate(tqdm(dataset)):
            if(len(x_batch) != BATCH_SIZE):
                continue
            with tf.GradientTape() as tape:

                x_batch = maybe_pad_sentences(x_batch)
                sl_b = [len(y) for y in y_batch]
                y_batch = maybe_pad_sentences(y_batch)
                h_b = encoder(
                    tf.convert_to_tensor(x_batch, dtype='float32'), is_training=True
                )

                o = tf.convert_to_tensor([0] * BATCH_SIZE, dtype='float32')
                logits_seq = []
                for idx in range(y_batch.shape[1] + 1):
                    # use teacher forcing
                    # cell_state_b = (
                    #     h_b,
                    #     decoder_cell.lstm_cell2.zero_state(
                    #         BATCH_SIZE, dtype='float32'),
                    #     decoder_cell.lstm_cell3.zero_state(
                    #         BATCH_SIZE, dtype='float32'),
                    # )
                    cell_state_b = h_b
                    w_logits, cell_state_b = decoder_cell(
                        o, cell_state_b, is_training=True
                    )

                    if (idx < y_batch.shape[1]):
                        o = tf.convert_to_tensor(y_batch[:, idx])
                        logits_seq.append(w_logits)

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
            optimizer.apply_gradients(zip(grads, model_vars))

            if batch % 100 == 0:
                sample_xs = random.sample(trainX, BATCH_SIZE)
                sample_xs = maybe_pad_sentences(sample_xs)
                h_b = encoder(
                    tf.convert_to_tensor(sample_xs, dtype='float32'),
                    is_training=False
                )
                outputs = []
                o = tf.convert_to_tensor([0] * BATCH_SIZE, dtype='float32')
                for idx in range(MAX_TOKENS_TARG):
                    # use teacher forcing
                    # cell_state_b = (
                    #     h_b,
                    #     decoder_cell.lstm_cell2.zero_state(
                    #         BATCH_SIZE, dtype='float32'),
                    #     decoder_cell.lstm_cell3.zero_state(
                    #         BATCH_SIZE, dtype='float32'),
                    # )
                    cell_state_b = h_b
                    w_logits, cell_state_b = decoder_cell(
                        o, cell_state_b, is_training=False)
                    o = tf.nn.softmax(w_logits)
                    o = tf.math.argmax(o, axis=-1)
                    outputs.append(o.numpy())
                outputs = np.rollaxis(np.asarray(outputs), 0, 1)
                outputs = [
                    decode_sentence(sentence, lang.idx2word)
                    for sentence in outputs
                ]
                sample_pairs = random.sample(list(zip(sample_xs, outputs)), 5)
                for q, a in sample_pairs:
                    print("Q: ", decode_sentence(q, lang.idx2word))
                    print("A: ", a)

                print("loss: ", loss.numpy())
        if epoch % 5 == 0:
            print("Saving models...")
            save(encoder, optimizer, 'checkpoints/encoder')
            save(decoder_cell, optimizer, 'checkpoints/decoder')
