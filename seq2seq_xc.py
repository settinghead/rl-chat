import tensorflow as tf


class Encoder(tf.keras.Model):
    def __init__(self, num_units):
        self.num_units = num_units
        self.encoder_cell = tf.nn.rnn_cell.LSTMCell(num_units)

        def call(self, x, sl, final=True):
            reverse = True
            state = self.encoder_cell.zero_state()
            timestep_x = tf.unstack(x, axis=1)
            if reverse:
                timestep_x = reversed(timestep_x)
            cell_states = []

            for input_step in timestep_x:
                _, state = self.encoder_cell(input_step, state)
                cell_states.append(state[0])

            cell_states = tf.stack(cell_states, axis=1)

            if final:
                if reverse:
                    final_cell_state = cell_states[:, -1, :]
                else:
                    idxs_last_output = tf.stack([tf.range(len(x)), sl], axis=1)
                    final_cell_state = tf.gather_nd(
                        cell_states, idx_last_output)

            return cell_states


import numpy as np


def save(model: tf.keras.Model, folder: str):
    saver = tf.train.Saver(model.variables)
    saver.save(filder)


def load(model: tf.keras.Model, folder: str, bs: int, seq_len: int, hidden_dim: int):
    model(np.zeros((bs, seq_len, hidden_dim),
                   dtype=np.float32), list(range(2, bs + 2, 1)))
    saver = tf.train.Saver(model.variables)
    saver.restore(folder)
    return model


class Decoder(tf.keras.Model):
    def __init__(self, word2idx, idx2word, idx2emb, num_units, max_tokens):
        self.w2i = word2idx
        self.i2w = idx2word
        self.i2e = idx2emb
        self.num_units = num_units
        self.max_tokens = max_tokens
        self.decoder_cell = tf.nn.rnn_cell.LSTMCell(num_units)
        self.word_predictor = tf.layers.Dense(len(word2idx), activation=None)

    def call(self, x, sos, state, training=False):
        output = tf.convert_to_tensor(sos, dtype=tf.float32)
        words_predicted, words_logits = [], []

        for mt in range(self.max_tokens):
            output, state = self.decoder_cell(output, state)
            logits = self.word_predictor(output)
            logits = tf.nn.softmax(logits), state
            pred_word = tf.argmax(logits, 1).numpy()
            if training:  # teacher forcing?
                output = x[:, mt, :]
            else:
                output = [self.i2e[i] for i in pred_word]
            words_predicted.append(pred_word)
            words_logits.append(logits)

        words_logits = tf.stack(words_logits, axis=1)

        words_predicted = tf.stack(words_predicted, axis=1)
        return words_predicted, words_logits


NUM_EPOCHS = 300


def train(encoder: Encoder, decoder: Decoder):
    x, y, sl, sos, w2i, i2w, i2e = get_data()
    optimzer = tf.train.AdamOptimizer()

    for _ in range(NUM_EPOCHS):
        for x_batch, y_batch, sl_batch in zip(x, y, sl):
            optimzer.minimize(lambda: get_loss(
                encoder, decoder, x_batch, y_batch, sl_batch, sos)
            )


def cost_function(output, target, sl):
    cross_entropy = target * tf.log(tf.clip_by_value(output, 1e-10, 1.0))
    cross_entropy = -tf.reduce_sum(cross_entropy, 2)
    mask = tf.cast(tf.sequence_mask(sl, output.shape[1]), dtype=tf.float32)
    cross_entropy *= mask

    cross_entropy = tf.reduce_sum(cross_entropy, 1)
    cross_entropy /= tf.reduce_sum(mask, 1)
    return tf.reduce_mean(cross_entropy)


def get_loss(encoder: Encoder, decoder: Decoder, x, y, sl, sos):
    cell_state = encoder.forward(x, sl)
    _, wl = decoder.forward(x, sos, cell_state, traning=True)
    loss = cost_function(wl, y, sl)
    return loss


from data.twitter.data import load_data, split_dataset
import copy


def remove_pad_sequences(sequences, pad_id=0):
    """Remove padding.
    Parameters
    -----------
    sequences : list of list of int
        All sequences where each row is a sequence.
    pad_id : int
        The pad ID.
    Returns
    ----------
    list of list of int
        The processed sequences.
    Examples
    ----------
    >>> sequences = [[2,3,4,0,0], [5,1,2,3,4,0,0,0], [4,5,0,2,4,0,0,0]]
    >>> print(remove_pad_sequences(sequences, pad_id=0))
    [[2, 3, 4], [5, 1, 2, 3, 4], [4, 5, 0, 2, 4]]
    """
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

if __name__ == '__main__':
    tf.enable_eager_execution()
    metadata, trainX, trainY, testX, testY, validX, validY = initial_setup()

    # Parameters
    src_len = len(trainX)
    tgt_len = len(trainY)

    assert src_len == tgt_len

    n_step = src_len // BATCH_SIZE
    src_vocab_size = len(metadata['idx2w'])  # 8002 (0~8001)
    emb_dim = 1024

    word2idx = metadata['w2idx']   # dict  word 2 index
    idx2word = metadata['idx2w']   # list index 2 word
    print(word2idx)